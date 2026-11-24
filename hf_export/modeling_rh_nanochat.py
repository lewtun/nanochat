"""Transformers implementation of the RH nanochat checkpoint architecture."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.nanochat.modeling_nanochat import NanoChatForCausalLM, NanoChatPreTrainedModel
from .configuration_rh_nanochat import RHNanoChatConfig


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.i = i
        self.nh, self.nkv = c.num_attention_heads, c.num_key_value_heads
        self.d = c.hidden_size // self.nh
        self.c_q = Linear(c.hidden_size, self.nh * self.d, bias=False)
        self.c_k = Linear(c.hidden_size, self.nkv * self.d, bias=False)
        self.c_v = Linear(c.hidden_size, self.nkv * self.d, bias=False)
        self.c_proj = Linear(c.hidden_size, c.hidden_size, bias=False)
        self.ve_gate = Linear(12, self.nkv, bias=False) if i % 2 == (c.num_hidden_layers - 1) % 2 else None

    def forward(self, x, ve, cos, sin, window, cache, attention_mask):
        b, t, _ = x.shape
        q = self.c_q(x).view(b, t, self.nh, self.d)
        k = self.c_k(x).view(b, t, self.nkv, self.d)
        v = self.c_v(x).view(b, t, self.nkv, self.d)
        if ve is not None:
            v = v + (3 * torch.sigmoid(self.ve_gate(x[..., :12]))).unsqueeze(-1) * ve.view(b, t, self.nkv, self.d)
        def rotate(a):
            a1, a2 = a.chunk(2, dim=-1)
            return torch.cat((a1 * cos + a2 * sin, -a1 * sin + a2 * cos), dim=-1)
        q, k = norm(rotate(q)) * 1.2, norm(rotate(k)) * 1.2
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        if cache is not None:
            k, v = cache.update(k, v, self.i)
        total = k.size(2)
        rows = torch.arange(total - t, total, device=x.device)[:, None]
        cols = torch.arange(total, device=x.device)[None, :]
        mask = (cols <= rows) & (rows - cols <= window)
        if attention_mask is not None:
            mask = mask[None, None] & attention_mask[:, None, None, :total].bool()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.nh != self.nkv)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_fc = Linear(c.hidden_size, 4 * c.hidden_size, bias=False)
        self.c_proj = Linear(4 * c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.attn, self.mlp = Attention(c, i), MLP(c)

    def forward(self, x, ve, cos, sin, window, cache, mask):
        x = x + self.attn(norm(x), ve, cos, sin, window, cache, mask)
        return x + self.mlp(norm(x))


class RHNanoChatForCausalLM(NanoChatForCausalLM):
    config_class = RHNanoChatConfig
    _tied_weights_keys = {}
    _supports_static_cache = False

    def __init__(self, config):
        NanoChatPreTrainedModel.__init__(self, config)
        c = config
        padded = ((c.vocab_size + 63) // 64) * 64
        self.transformer = nn.ModuleDict({"wte": nn.Embedding(padded, c.hidden_size),
            "h": nn.ModuleList([Block(c, i) for i in range(c.num_hidden_layers)])})
        self.lm_head = Linear(c.hidden_size, padded, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(c.num_hidden_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(c.num_hidden_layers))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.zeros(1))
        d = c.num_key_value_heads * (c.hidden_size // c.num_attention_heads)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded, d) for i in range(c.num_hidden_layers) if i % 2 == (c.num_hidden_layers - 1) % 2})
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_output_embeddings(self):
        return self.lm_head

    def _reorder_cache(self, cache, beam_idx):
        cache.reorder_cache(beam_idx)
        if hasattr(cache, "rh_prev_embedding"):
            cache.rh_prev_embedding = cache.rh_prev_embedding.index_select(0, beam_idx.to(cache.rh_prev_embedding.device))
        return cache

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, cache=None, use_cache=None, labels=None, logits_to_keep=0, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if use_cache and cache is None:
            cache = DynamicCache()
        start = cache.get_seq_length() if cache is not None else 0
        b, t = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(start, start + t, device=input_ids.device).unsqueeze(0)
        x = norm(self.transformer.wte(input_ids))
        prev = getattr(cache, "rh_prev_embedding", None) if cache is not None else None
        if cache is not None:
            cache.rh_prev_embedding = x[:, -1:, :]
        previous = torch.cat((prev if prev is not None else torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        if attention_mask is not None:
            valid = attention_mask[:, start:start + t].unsqueeze(-1).to(x.dtype)
            x = x * valid
            previous = previous * torch.cat((attention_mask[:, start-1:start] if start else torch.zeros_like(attention_mask[:, :1]), attention_mask[:, start:start+t-1]), dim=1).unsqueeze(-1).to(x.dtype)
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[..., :24]))
        x = x + gate * previous
        x0 = x
        d = self.config.hidden_size // self.config.num_attention_heads
        inv = 1.0 / (100000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        freqs = position_ids.float().unsqueeze(-1) * inv
        cos, sin = freqs.cos().to(x.dtype).unsqueeze(2), freqs.sin().to(x.dtype).unsqueeze(2)
        short = -(-self.config.max_position_embeddings // 4 // 128) * 128
        backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None
            window = short if self.config.window_pattern.upper()[i % len(self.config.window_pattern)] == 'S' and i != len(self.transformer.h) - 1 else self.config.max_position_embeddings
            x = block(x, ve, cos, sin, window, cache, attention_mask)
            if i == len(self.transformer.h) // 2:
                backout = x
        if backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * backout
        x = norm(x)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)
        loss = None if labels is None else F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, cache=cache)

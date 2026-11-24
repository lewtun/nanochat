"""CPU parity and serialization tests for the Transformers exporter."""
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from hf_export.convert import convert, convert_tokenizer
from hf_export.configuration_rh_nanochat import RHNanoChatConfig
from hf_export.modeling_rh_nanochat import RHNanoChatForCausalLM


torch.set_num_threads(1)


def models(depth):
    c = GPTConfig(sequence_len=256, vocab_size=270, n_layer=depth, n_head=4, n_kv_head=2, n_embd=32, window_pattern='SL')
    original = GPT(c)
    original.init_weights()
    torch.manual_seed(12)
    with torch.no_grad():
        for p in original.parameters():
            p.uniform_(-0.15, 0.15)
    hc = RHNanoChatConfig(vocab_size=c.vocab_size, hidden_size=c.n_embd, intermediate_size=4*c.n_embd, num_hidden_layers=c.n_layer, num_attention_heads=c.n_head, num_key_value_heads=c.n_kv_head, max_position_embeddings=c.sequence_len, window_pattern=c.window_pattern, bos_token_id=None, eos_token_id=None, pad_token_id=None)
    hf = RHNanoChatForCausalLM(hc)
    hf.load_state_dict(original.state_dict())
    return original.eval(), hf.eval()


@pytest.mark.parametrize('depth,length', [(1, 17), (2, 150), (4, 33)])
def test_logits_and_cache(depth, length):
    orig, hf = models(depth)
    ids = torch.randint(0, 270, (1, length))
    with torch.no_grad():
        expected = orig(ids)
        result = hf(ids).logits
        pre = hf(ids[:, :-1], use_cache=True)
        last = hf(ids[:, -1:], cache=pre.cache, use_cache=True).logits
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(last, expected[:, -1:], atol=2e-6, rtol=2e-5)


def test_greedy_engine():
    orig, hf = models(2)
    class DummyTokenizer:
        def encode_special(self, token): return 999
        def get_bos_token_id(self): return 999
    ids = [1, 5, 9, 7]
    engine = Engine(orig, DummyTokenizer())
    expected = [row[0][0] for row in engine.generate(ids, max_tokens=8, temperature=0)]
    actual = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):].tolist()
    assert actual == expected


def test_convert_roundtrip(tmp_path):
    tokenizer = RustBPETokenizer.train_from_iterator(iter(['theorem foo : Nat := 42\n' * 100, '∀ x : ℕ, x = x\n' * 100]), 270)
    tokdir, ckpt, out = (tmp_path / name for name in ('tok', 'ckpt', 'out'))
    tokdir.mkdir(); ckpt.mkdir()
    tokenizer.save(tokdir)
    orig, _ = models(2)
    torch.save(orig.state_dict(), ckpt / 'model_000001.pt')
    (ckpt / 'meta_000001.json').write_text(json.dumps({'model_config': vars(orig.config)}))
    plots = tmp_path / 'plots'
    plots.mkdir()
    (plots / 'isoflop.png').write_bytes(b'example plot')
    results = tmp_path / 'results.md'
    results.write_text('## Scaling-law results\n\n![isoFLOP](assets/isoflop.png)\n\n| N | D |\n|---|---|\n| 1 | 2 |\n')
    convert(ckpt, tokdir, out, results_markdown=results, assets_dir=plots)
    assert '![isoFLOP](assets/isoflop.png)' in (out / 'README.md').read_text()
    assert (out / 'assets/isoflop.png').read_bytes() == b'example plot'
    assert all((out / name).exists() for name in ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "README.md"))
    model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    fast = AutoTokenizer.from_pretrained(out)
    text = 'theorem foo : ℕ := 42\n∀ x, x = x'
    assert fast.encode(text, add_special_tokens=False) == tokenizer.encode(text)
    for s in SPECIAL_TOKENS:
        assert fast.encode(s, add_special_tokens=False) == [tokenizer.encode_special(s)]
        assert fast.decode([tokenizer.encode_special(s)]) == s
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(ids).logits, orig(ids), atol=2e-6, rtol=2e-5)
    assert model.generate(ids, max_new_tokens=2).shape == (1, 5)


def test_climbmix(tmp_path):
    path = Path(os.environ.get('AM_LOCAL', '/nonexistent')) / 'rh-corpus/tok-climbmix'
    if not (path / 'tokenizer.pkl').exists():
        pytest.skip('ClimbMix tokenizer unavailable')
    original = RustBPETokenizer.from_directory(path)
    convert_tokenizer(path, tmp_path)
    fast = AutoTokenizer.from_pretrained(tmp_path)
    text = 'theorem foo : ℕ := 42\n∀ x, x = x'
    assert fast.encode(text, add_special_tokens=False) == original.encode(text)
    assert fast.decode(original.encode(text)) == text


def test_bfloat16(monkeypatch):
    import nanochat.gpt as gpt_module
    original, model = models(2)
    monkeypatch.setattr(gpt_module, "COMPUTE_DTYPE", torch.bfloat16)
    original.to(torch.bfloat16)
    original.cos, original.sin = original._precompute_rotary_embeddings(original.rotary_seq_len, 8)
    model.to(torch.bfloat16)
    ids = torch.randint(0, 270, (1, 25))
    with torch.no_grad():
        expected = original(ids)
        result = model(ids).logits
        pre = model(ids[:, :-1], use_cache=True)
        last = model(ids[:, -1:], cache=pre.cache, use_cache=True).logits
    torch.testing.assert_close(result, expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(last, result[:, -1:], atol=0.02, rtol=0.05)


def test_real_lean_climbmix(tmp_path):
    root = Path(os.environ.get('AM_LOCAL', '/nonexistent')) / 'rh-corpus'
    path = root / 'tok-climbmix'
    sources = [root / 'extract/lean/src/Main.lean', root / 'extract/lean/src/Statements.lean']
    if not (path / 'tokenizer.pkl').exists() or not all(p.exists() for p in sources):
        pytest.skip('Lean corpus or ClimbMix tokenizer unavailable')
    original = RustBPETokenizer.from_directory(path)
    convert_tokenizer(path, tmp_path)
    fast = AutoTokenizer.from_pretrained(tmp_path)
    for source in sources:
        text = source.read_text()
        ids = original.encode(text)
        assert fast.encode(text, add_special_tokens=False) == ids
        assert fast.decode(ids) == original.decode(ids)


def test_beam_generation_cache():
    _, model = models(2)
    ids = torch.tensor([[1, 5, 9, 7]])
    cached = model.generate(ids, max_new_tokens=8, num_beams=3, use_cache=True)
    uncached = model.generate(ids, max_new_tokens=8, num_beams=3, use_cache=False)
    assert torch.equal(cached, uncached)

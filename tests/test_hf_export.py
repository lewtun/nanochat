"""CPU parity and serialization tests for the Transformers exporter."""
import json
import hashlib
import sys
import random
import shutil
import string
from types import SimpleNamespace
import os
from pathlib import Path

import pytest
import torch
pytest.importorskip("transformers")
pytest.importorskip("blobfile")
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
import tiktoken
from hf_export.convert import convert, convert_tokenizer
from hf_export.configuration_rh_nanochat import RHNanoChatConfig
from hf_export.modeling_rh_nanochat import RHNanoChatForCausalLM


@pytest.fixture(autouse=True)
def cpu_defaults(monkeypatch):
    import nanochat.gpt as gpt
    import nanochat.engine as engine
    import nanochat.flash_attention as attention
    monkeypatch.setattr(gpt, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(engine, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(attention, 'USE_FA3', False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def train_tokenizer(texts, vocab_size):
    source = RustBPETokenizer.train_from_iterator(texts, vocab_size - 1)
    enc = source.enc
    special = dict(enc._special_tokens)
    special['<|eos|>'] = enc.n_vocab
    return RustBPETokenizer(tiktoken.Encoding(name='test-eos', pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks, special_tokens=special), '<|bos|>')


def models(depth, sequence_len=256, window_pattern='SL'):
    c = GPTConfig(sequence_len=sequence_len, vocab_size=270, n_layer=depth, n_head=4, n_kv_head=2, n_embd=32, window_pattern=window_pattern)
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
    tokenizer = train_tokenizer(iter(['theorem foo : Nat := 42\n' * 100, '∀ x : ℕ, x = x\n' * 100]), 270)
    tokdir, ckpt, out = (tmp_path / name for name in ('tok', 'ckpt', 'out'))
    tokdir.mkdir(); ckpt.mkdir()
    tokenizer.save(tokdir)
    orig, _ = models(2)
    torch.save(orig.state_dict(), ckpt / 'model_000001.pt')
    (ckpt / 'meta_000001.json').write_text(json.dumps({'model_config': vars(orig.config), 'tokenizer_sha256': hashlib.sha256((tokdir / 'tokenizer.pkl').read_bytes()).hexdigest()}))
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
    fast = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    text = 'theorem foo : ℕ := 42\n∀ x, x = x'
    assert fast.encode(text, add_special_tokens=False) == tokenizer.encode(text)
    assert fast.encode(text) == [tokenizer.get_bos_token_id()] + tokenizer.encode(text)
    assert fast.bos_token_id == tokenizer.get_bos_token_id()
    assert fast.pad_token_id == fast.eos_token_id == tokenizer.encode_special('<|eos|>')
    assert fast.bos_token_id != fast.eos_token_id
    assert fast.padding_side == 'left'
    assert fast.model_max_length == orig.config.sequence_len
    for s in SPECIAL_TOKENS + ['<|eos|>']:
        assert fast.encode(s, add_special_tokens=False) == [tokenizer.encode_special(s)]
        assert fast.decode([tokenizer.encode_special(s)]) == s
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(ids).logits, orig(ids), atol=2e-6, rtol=2e-5)
    assert model.generation_config.eos_token_id == tokenizer.encode_special('<|eos|>')
    assert model.generation_config.pad_token_id == tokenizer.encode_special('<|eos|>')
    from transformers import LogitsProcessor, LogitsProcessorList
    class ForceEOS(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(float('-inf'))
            scores[:, tokenizer.encode_special('<|eos|>')] = 0
            return scores
    generated = model.generate(ids, max_new_tokens=5, logits_processor=LogitsProcessorList([ForceEOS()]))
    assert generated.shape == (1, 4) and generated[0, -1] == tokenizer.encode_special('<|eos|>')
    repo = tmp_path / 'repo'; run = repo / 'run'
    shutil.copytree(out, run)
    for name in ('configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        shutil.copyfile(out / name, repo / name)
    submodel = AutoModelForCausalLM.from_pretrained(repo, subfolder='run', trust_remote_code=True)
    torch.testing.assert_close(submodel(ids).logits, model(ids).logits)


def test_climbmix(tmp_path):
    path = Path(os.environ.get('RH_TEST_TOKENIZER', '/nonexistent'))
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
    torch.testing.assert_close(last, expected[:, -1:], atol=0.002, rtol=0.01)


def test_real_lean_climbmix(tmp_path):
    path = Path(os.environ.get('RH_TEST_TOKENIZER', '/nonexistent'))
    root = Path(os.environ.get('RH_TEST_LEAN', '/nonexistent'))
    sources = [root / 'Main.lean', root / 'Statements.lean']
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


@pytest.mark.parametrize('depth,pattern', [(2, 'S'), (3, 'SSSL')])
def test_long_window_and_generation(depth, pattern):
    orig, model = models(depth, 640, pattern)
    ids = torch.randint(0, 270, (1, 400))
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, orig(ids), atol=2e-6, rtol=2e-5)
    class Tokenizer:
        def encode_special(self, token): return 999
        def get_bos_token_id(self): return 999
    expected = [row[0][0] for row in Engine(orig, Tokenizer()).generate(ids[0].tolist(), max_tokens=4, temperature=0)]
    actual = model.generate(ids, max_new_tokens=4, do_sample=False)[0, 400:].tolist()
    assert actual == expected


def test_padding_cache_reorder_and_loss():
    orig, model = models(2)
    a, b = torch.tensor([5, 6, 7]), torch.tensor([4, 2, 8, 9, 10])
    batch = torch.tensor([[0, 0, *a.tolist()], b.tolist()])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    pos = mask.cumsum(-1).sub(1).clamp(min=0)
    with torch.no_grad():
        result = model(batch, attention_mask=mask, position_ids=pos).logits
        torch.testing.assert_close(result[0, -3:], orig(a[None])[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(result[1], orig(b[None])[0], atol=2e-6, rtol=2e-5)
        cache = model(batch, attention_mask=mask, position_ids=pos).cache
        model._reorder_cache(cache, torch.tensor([1, 0]))
        next_ids = torch.tensor([[11], [12]])
        masks = torch.cat((mask[[1, 0]], torch.ones(2, 1, dtype=torch.long)), dim=1)
        positions = masks.sum(-1, keepdim=True) - 1
        decoded = model(next_ids, cache=cache, attention_mask=masks, position_ids=positions).logits
        for i, seq in enumerate((torch.cat((b, next_ids[0])), torch.cat((a, next_ids[1])))):
            torch.testing.assert_close(decoded[i, 0], orig(seq[None])[0, -1], atol=2e-6, rtol=2e-5)
        labels = torch.tensor([[1, 2, 3, 4]])
        out = model(labels, labels=labels, use_cache=False)
        loss = torch.nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, 270), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(out.loss, loss)


def test_prompt_lookup_and_cache_crop():
    _, model = models(2)
    prompt = torch.tensor([[3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]])
    plain = model.generate(prompt, max_new_tokens=8, do_sample=False)
    assisted = model.generate(prompt, max_new_tokens=8, do_sample=False, prompt_lookup_num_tokens=4)
    assert torch.equal(plain, assisted)
    pre = model(prompt)
    pre.cache.crop(-7)
    result = model(prompt[:, 5:8], cache=pre.cache).logits
    torch.testing.assert_close(result, model(prompt[:, :8], use_cache=False).logits[:, 5:], atol=2e-6, rtol=2e-5)


def test_upload_guard(tmp_path, monkeypatch):
    import hf_export.upload as upload
    for name in ('config.json', 'tokenizer.json', 'model.safetensors', 'configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        (tmp_path / name).write_text('test')
    calls = []
    class API:
        def repo_info(self, *args, **kwargs): return SimpleNamespace(private=self.private)
        def file_exists(self, *args, **kwargs): return False
        def upload_file(self, **kwargs): calls.append(kwargs)
        def upload_folder(self, **kwargs): calls.append(kwargs)
    api = API(); monkeypatch.setattr(upload, 'HfApi', lambda: api)
    monkeypatch.setattr(sys, 'argv', ['upload', str(tmp_path), 'owner/repo', '--path-in-repo', 'run'])
    api.private = False
    with pytest.raises(ValueError, match='public'):
        upload.main()
    assert not calls
    api.private = True
    upload.main()
    assert calls[-1]['path_in_repo'] == 'run'
    assert '*.pt' in calls[-1]['ignore_patterns'] and '*.pkl' in calls[-1]['ignore_patterns']
    assert {c['path_in_repo'] for c in calls[:-1]} == {'configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'}


def test_portable_tokenizer(tmp_path):
    rng = random.Random(42)
    words = [''.join(rng.choices(string.ascii_lowercase, k=9)) for _ in range(400)]
    texts = [f'lemma {words[i % 400]} : ℕ := {i * 173}  -- αβ\r\n  ∀ x, x = x\n\n' for i in range(2400)]
    tokenizer = train_tokenizer(iter(texts), 640)
    assert tokenizer.get_vocab_size() >= 512
    tokdir = tmp_path / 'source'; tokdir.mkdir(); tokenizer.save(tokdir)
    out = tmp_path / 'fast'; out.mkdir()
    convert_tokenizer(tokdir, out, 640)
    fast = AutoTokenizer.from_pretrained(out)
    for text in texts[::29] + [' 12345\r\n∀ x : ℕ, x = 0\n   \t 99', 'αβ 😀\n  123456789']:
        ids = tokenizer.encode(text)
        assert fast.encode(text, add_special_tokens=False) == ids
        assert fast.encode(text) == [tokenizer.get_bos_token_id()] + ids
        assert fast.decode(ids) == tokenizer.decode(ids)
    for token in SPECIAL_TOKENS + ['<|eos|>']:
        assert fast.encode(token, add_special_tokens=False) == [tokenizer.encode_special(token)]


def test_conversion_provenance_and_strict_weights(tmp_path):
    tokenizer = train_tokenizer(iter(['theorem foo : Nat := 42\n' * 100, '∀ x : ℕ, x = x\n' * 100]), 270)
    tokdir, ckpt, out = (tmp_path / name for name in ('tok', 'ckpt', 'out'))
    tokdir.mkdir(); ckpt.mkdir(); tokenizer.save(tokdir)
    orig, _ = models(4, 640, 'S')
    state = orig.state_dict()
    for key in state:
        if key.startswith('transformer.wte') or key.startswith('value_embeds.'):
            state[key] = state[key].bfloat16()
    torch.save(state, ckpt / 'model_000001.pt')
    torch.save(state, ckpt / 'model_000002.pt')
    (ckpt / 'model_final.pt').write_text('ignore')
    sha = hashlib.sha256((tokdir / 'tokenizer.pkl').read_bytes()).hexdigest()
    for step in (1, 2):
        (ckpt / f'meta_{step:06d}.json').write_text(json.dumps({'model_config': vars(orig.config), 'tokenizer_sha256': sha}))
    with pytest.raises(ValueError, match='conflicts'):
        convert(ckpt, tokdir, out, tokenizer_sha256='0' * 64)
    convert(ckpt, tokdir, out)
    config = json.loads((out / 'config.json').read_text())
    assert (config['checkpoint_step'], config['window_pattern'], config['max_position_embeddings'], config['tokenizer_sha256']) == (2, 'S', 640, sha)
    model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32)
    orig.load_state_dict({k: v.float() for k, v in state.items()}, assign=True)
    ids = torch.randint(0, 270, (1, 400))
    torch.testing.assert_close(model(ids).logits, orig(ids), atol=2e-6, rtol=2e-5)
    with pytest.raises(FileExistsError):
        convert(ckpt, tokdir, out)
    state.pop('lm_head.weight'); torch.save(state, ckpt / 'model_000002.pt')
    with pytest.raises(RuntimeError, match='Missing key'):
        convert(ckpt, tokdir, out, overwrite=True)
    assert (out / 'model.safetensors').exists()


def test_config_and_cache_contract():
    _, model = models(2)
    ids = torch.tensor([[1, 2, 3]])
    pre = model(ids)
    assert pre.cache is not None
    with pytest.warns(FutureWarning):
        cached = model(torch.tensor([[4]]), past_key_values=pre.cache).logits
    torch.testing.assert_close(cached, model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[:, -1:])
    assert isinstance(model(ids, return_dict=False), tuple)
    with pytest.raises(ValueError, match='attention'):
        model(ids, attention_mask=torch.ones(1, 1, 3, 3))
    with pytest.raises(ValueError, match='window_pattern'):
        RHNanoChatConfig(hidden_size=32, intermediate_size=128, window_pattern='X')


def test_tokenizer_eos_and_config(tmp_path):
    tokenizer = train_tokenizer(iter(['∀ x : ℕ, theorem := 12345\n' * 100]), 280)
    source = tmp_path / 'source'; source.mkdir(); tokenizer.save(source)
    output = tmp_path / 'out'; output.mkdir()
    convert_tokenizer(source, output, 640)
    fast = AutoTokenizer.from_pretrained(output)
    batched = fast(['∀ x', '∀'], padding=True)
    assert batched['input_ids'][1][0] == fast.pad_token_id
    assert batched['attention_mask'][1][0] == 0
    assert fast.eos_token_id == tokenizer.encode_special('<|eos|>')
    assert fast.eos_token_id != tokenizer.get_bos_token_id()


def test_reset_and_output_contract():
    _, model = models(2)
    ids = torch.tensor([[1, 2, 3, 4]])
    pre = model(ids)
    pre.cache.reset()
    reset = model(ids, cache=pre.cache).logits
    torch.testing.assert_close(reset, model(ids, use_cache=False).logits)
    selected = model(ids, logits_to_keep=torch.tensor([0, 2]), use_cache=False).logits
    torch.testing.assert_close(selected, model(ids, use_cache=False).logits[:, [0, 2]])
    with pytest.raises(ValueError, match='output_hidden_states'):
        model(ids, output_hidden_states=True)

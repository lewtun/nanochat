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
pytest.importorskip("transformers", reason="Transformers export tests require Python 3.11 and transformers 5.24")
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
    (ckpt / 'meta_000001.json').write_text(json.dumps({'model_config': vars(orig.config), 'tokenizer_sha256': hashlib.sha256((tokdir / 'tokenizer.pkl').read_bytes()).hexdigest(), 'user_config': {'run': 'test-run'}, 'val_bpb': .5}))
    plots = tmp_path / 'plots'
    plots.mkdir()
    (plots / 'isoflop.png').write_bytes(b'example plot')
    results = tmp_path / 'results.md'
    results.write_text('## Scaling-law results\n\n![isoFLOP](assets/isoflop.png)\n\n| N | D |\n|---|---|\n| 1 | 2 |\n')
    (tmp_path / 'selected_model.json').write_text(json.dumps({'run_name': 'test-run', 'final_validation_bpb': .5}))
    convert(ckpt, tokdir, out, results_markdown=results, assets_dir=plots)
    assert 'license: apache-2.0' in (out / 'README.md').read_text()
    assert 'license terms have not been established' not in (out / 'README.md').read_text()
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


def test_real_tokenizer(tmp_path):
    path = Path(os.environ.get('RH_TEST_TOKENIZER', '/nonexistent'))
    if not (path / 'tokenizer.pkl').exists():
        pytest.skip('Set RH_TEST_TOKENIZER to the EOS tokenizer directory')
    original = RustBPETokenizer.from_directory(path)
    if '<|eos|>' not in original.enc._special_tokens:
        pytest.skip('Real tokenizer has no distinct EOS token')
    convert_tokenizer(path, tmp_path)
    fast = AutoTokenizer.from_pretrained(tmp_path)
    text = 'theorem foo : ℕ := 42\n∀ x, x = x'
    assert fast.encode(text, add_special_tokens=False) == original.encode(text)
    assert fast.decode(original.encode(text)) == text
    assert fast.encode(text) == [original.get_bos_token_id()] + original.encode(text)
    assert fast.eos_token_id == original.encode_special('<|eos|>')


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


def test_real_lean_tokenizer(tmp_path):
    path = Path(os.environ.get('RH_TEST_TOKENIZER', '/nonexistent'))
    root = Path(os.environ.get('RH_TEST_LEAN', '/nonexistent'))
    sources = [root / 'Main.lean', root / 'Statements.lean']
    if not (path / 'tokenizer.pkl').exists() or not all(p.exists() for p in sources):
        pytest.skip('Set RH_TEST_TOKENIZER and RH_TEST_LEAN for real-data parity')
    original = RustBPETokenizer.from_directory(path)
    if '<|eos|>' not in original.enc._special_tokens:
        pytest.skip('Real tokenizer has no distinct EOS token')
    convert_tokenizer(path, tmp_path)
    fast = AutoTokenizer.from_pretrained(tmp_path)
    for source in sources:
        text = source.read_text()
        ids = original.encode(text)
        assert fast.encode(text, add_special_tokens=False) == ids
        assert fast.decode(ids) == original.decode(ids)
        assert fast.encode(text) == [original.get_bos_token_id()] + ids
        assert fast.eos_token_id == original.encode_special('<|eos|>')


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
    replacement = torch.tensor([[19, 20, 21]])
    model(replacement, cache=pre.cache)
    result = model(torch.tensor([[22]]), cache=pre.cache).logits
    fresh = torch.cat((prompt[:, :5], replacement, torch.tensor([[22]])), dim=1)
    torch.testing.assert_close(result, model(fresh, use_cache=False).logits[:, -1:], atol=2e-6, rtol=2e-5)


def test_upload_guard(tmp_path, monkeypatch, capsys):
    import fnmatch
    import hf_export.upload as upload
    local = tmp_path / 'local'; local.mkdir()
    for name in ('config.json', 'tokenizer.json', 'model.safetensors', 'configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        (local / name).write_text('test')
    (local / 'README.md').write_text('new card')
    (local / 'assets').mkdir()
    (local / 'assets' / 'new.png').write_text('new image')
    remote = {'configuration_rh_nanochat.py': 'test', 'modeling_rh_nanochat.py': 'test', 'README.md': 'hand-edited', 'runA/README.md': 'hand-edited subfolder',
              'runA/config.json': 'old', 'runA/model.safetensors': 'weights', 'runA/assets/old.png': 'old',
              'runB/config.json': 'old', 'runB/model.safetensors': 'weights',
              'model.safetensors': 'old', 'assets/old.png': 'old'}
    calls = []
    class API:
        private = False
        def repo_info(self, *args, **kwargs): return SimpleNamespace(private=self.private, sha='revision')
        def file_exists(self, repo, path, **kwargs):
            assert kwargs['revision'] == 'revision'
            return path in remote
        def list_repo_files(self, *args, **kwargs):
            assert kwargs['revision'] == 'revision'
            return list(remote)
        def upload_file(self, **kwargs): calls.append(kwargs)
        def upload_folder(self, **kwargs): calls.append(kwargs)
    def download(repo, name, **kwargs):
        assert kwargs['revision'] == 'revision'
        p = tmp_path / 'download'; p.write_text(remote[name]); return str(p)
    api = API(); monkeypatch.setattr(upload, 'HfApi', lambda: api)
    monkeypatch.setattr(upload, 'hf_hub_download', download)
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--path-in-repo', './runA'])
    with pytest.raises(ValueError, match='public'): upload.main()
    api.private = True
    with pytest.raises(FileExistsError): upload.main()
    remote['modeling_rh_nanochat.py'] = 'different'
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--overwrite'])
    with pytest.raises(ValueError, match='differs'): upload.main()
    remote['modeling_rh_nanochat.py'] = 'test'
    upload.main()
    assert len(calls) == 1 and calls[-1]['path_in_repo'] == 'runA'
    assert calls[-1]['delete_patterns'] == ['model.safetensors']
    assert '*.pt' in calls[-1]['ignore_patterns']
    assert 'README.md' in calls[-1]['ignore_patterns']
    assert 'assets/**' in calls[-1]['ignore_patterns']
    assert 'runA/assets/old.png' in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite'])
    upload.main()
    patterns = calls[-1]['delete_patterns']
    assert 'README.md' in calls[-1]['ignore_patterns']
    deleted = {key for key in remote if any(fnmatch.fnmatchcase(key, pattern) for pattern in patterns)}
    assert deleted == {'model.safetensors'}
    assert 'assets/**' in calls[-1]['ignore_patterns']
    assert 'assets/old.png' in capsys.readouterr().out
    assert 'runA/model.safetensors' not in deleted and 'runB/model.safetensors' not in deleted
    from huggingface_hub.utils import filter_repo_objects
    assert list(filter_repo_objects(['README.md', 'assets/new.png', 'assets/nested/new.png', 'config.json'],
                                    ignore_patterns=calls[-1]['ignore_patterns'])) == ['config.json']
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite', '--replace-readme'])
    upload.main()
    assert 'README.md' not in calls[-1]['ignore_patterns']
    assert 'assets/**' not in calls[-1]['ignore_patterns']
    deleted = {key for key in remote if any(fnmatch.fnmatchcase(key, pattern) for pattern in calls[-1]['delete_patterns'])}
    assert deleted == {'model.safetensors', 'assets/old.png'}
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite', '--keep-readme', '--replace-readme'])
    with pytest.raises(SystemExit): upload.main()
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite', '--replace-readme'])
    (local / 'README.md').unlink()
    with pytest.raises(ValueError, match='local README.md is missing'): upload.main()
    del remote['README.md']
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite'])
    upload.main()
    assert 'README.md' not in calls[-1]['ignore_patterns']
    assert 'assets/**' not in calls[-1]['ignore_patterns']
    monkeypatch.setattr(sys, 'argv', ['upload', str(local), 'owner/repo', '--overwrite'])
    remote['modeling_rh_nanochat.py'] = 'different'
    with pytest.raises(ValueError, match='differs'): upload.main()


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
    meta_file = ckpt / 'meta_000002.json'
    meta = json.loads(meta_file.read_text())
    meta['tokenizer_sha256'] = '0' * 64; meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='does not match'): convert(ckpt, tokdir, out)
    assert not out.exists()
    del meta['tokenizer_sha256']; meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='no tokenizer_sha256'): convert(ckpt, tokdir, out)
    with pytest.raises(ValueError, match='does not match'): convert(ckpt, tokdir, out, tokenizer_sha256='0' * 64)
    assert not out.exists()
    for protected in (ckpt, tokdir, tmp_path):
        with pytest.raises(ValueError, match='must not contain'): convert(ckpt, tokdir, protected, overwrite=True)
    assert (ckpt / 'model_000002.pt').exists() and (tokdir / 'tokenizer.pkl').exists()
    convert(ckpt, tokdir, out, tokenizer_sha256=sha)
    shutil.rmtree(out)
    meta['tokenizer_sha256'] = sha; meta_file.write_text(json.dumps(meta))
    convert(ckpt, tokdir, out)
    config = json.loads((out / 'config.json').read_text())
    assert (config['checkpoint_step'], config['window_pattern'], config['max_position_embeddings'], config['tokenizer_sha256']) == (2, 'S', 640, sha)
    model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32, attn_implementation='sdpa', max_position_embeddings=1024)
    assert model.config.training_sequence_len == 640 and model.config.max_position_embeddings == 1024
    assert model.config.base_model_tp_plan is None
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


def test_export_safety_and_asset_validation(tmp_path):
    tokenizer = train_tokenizer(iter(['theorem foo : Nat := 42\n' * 100, '∀ x : ℕ, x = x\n' * 100]), 270)
    tokdir, ckpt, out, assets = (tmp_path / name for name in ('tok', 'ckpt', 'out', 'assets'))
    tokdir.mkdir(); ckpt.mkdir(); assets.mkdir(); tokenizer.save(tokdir)
    original, _ = models(2)
    torch.save(original.state_dict(), ckpt / 'model_000001.pt')
    sha = hashlib.sha256((tokdir / 'tokenizer.pkl').read_bytes()).hexdigest()
    (ckpt / 'meta_000001.json').write_text(json.dumps({'model_config': vars(original.config), 'tokenizer_sha256': sha, 'user_config': {'run': 'test-run'}, 'val_bpb': .5}))
    unrelated = tmp_path / 'unrelated'; unrelated.mkdir(); (unrelated / 'precious').write_text('keep')
    with pytest.raises(ValueError, match='not an RH nanochat export'):
        convert(ckpt, tokdir, unrelated, overwrite=True)
    assert (unrelated / 'precious').read_text() == 'keep'
    link = tmp_path / 'link'; link.symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        convert(ckpt, tokdir, link, overwrite=True)
    results = tmp_path / 'results.md'
    (tmp_path / 'selected_model.json').write_text(json.dumps({'run_name': 'test-run', 'final_validation_bpb': .5}))
    (assets / 'notes.txt').write_text('not an image')
    for text in ('![x](assets/notes.txt)', '![x](assets/../x.png)', '![x](assets/missing.png)', '<img src="assets/a.png">'):
        results.write_text(text)
        with pytest.raises(ValueError): convert(ckpt, tokdir, out, results_markdown=results, assets_dir=assets)
        assert not out.exists()
    (assets / 'a.png').write_bytes(b'image'); (assets / 'unreferenced.png').write_bytes(b'no')
    results.write_text('![x](assets/a.png)')
    (tmp_path / 'selected_model.json').write_text(json.dumps({'run_name': 'wrong-run', 'final_validation_bpb': .5}))
    with pytest.raises(ValueError, match='does not match checkpoint'):
        convert(ckpt, tokdir, out, results_markdown=results, assets_dir=assets)
    (tmp_path / 'selected_model.json').write_text(json.dumps({'run_name': 'test-run', 'final_validation_bpb': .6}))
    with pytest.raises(ValueError, match='does not match checkpoint'):
        convert(ckpt, tokdir, out, results_markdown=results, assets_dir=assets)
    (tmp_path / 'selected_model.json').write_text(json.dumps({'run_name': 'test-run', 'final_validation_bpb': .5}))
    convert(ckpt, tokdir, out, results_markdown=results, assets_dir=assets)
    assert (out / 'assets/a.png').exists() and not (out / 'assets/unreferenced.png').exists()
    (out / 'assets/stale.png').write_bytes(b'stale')
    convert(ckpt, tokdir, out, results_markdown=results, assets_dir=assets, overwrite=True)
    assert not (out / 'assets/stale.png').exists()


def test_config_validation_and_static_cache():
    from transformers.cache_utils import StaticCache
    assert RHNanoChatConfig().intermediate_size == 3072
    for kwargs in ({'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000}},
                   {'rms_norm_eps': 1e-6}, {'final_logit_softcapping': 30.0}):
        with pytest.raises(ValueError): RHNanoChatConfig(**kwargs)
    _, model = models(2)
    cache = StaticCache(config=model.config, max_cache_len=32)
    with pytest.raises(ValueError, match='Static cache'):
        model(torch.tensor([[1, 2, 3]]), cache=cache)

"""Convert a trusted nanochat checkpoint and its exact training tokenizer."""
import argparse
import hashlib
import json
import pickle
import re
import shutil
import tempfile
from pathlib import Path

import torch
from tiktoken.load import dump_tiktoken_bpe
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter
from .configuration_rh_nanochat import RHNanoChatConfig
from .modeling_rh_nanochat import RHNanoChatForCausalLM


def convert_tokenizer(tokenizer_dir, output_dir, sequence_len=2048):
    # Pickle can execute arbitrary code. Only load trusted files.
    with open(Path(tokenizer_dir) / 'tokenizer.pkl', 'rb') as f:
        enc = pickle.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        ranks = str(Path(tmp) / 'ranks.tiktoken')
        dump_tiktoken_bpe(enc._mergeable_ranks, ranks)
        backend = TikTokenConverter(vocab_file=ranks, pattern=enc._pat_str).converted()
    specials = sorted(enc._special_tokens.items(), key=lambda item: item[1])
    for token, expected in specials:
        backend.add_special_tokens([token])
        if backend.token_to_id(token) != expected:
            raise ValueError(f'Special token ID mismatch for {token}')
    if '<|eos|>' not in enc._special_tokens:
        raise ValueError('Tokenizer must contain a distinct <|eos|> token')
    bos = enc._special_tokens['<|bos|>']
    if enc._special_tokens['<|eos|>'] == bos:
        raise ValueError('BOS and EOS must have distinct token IDs')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<|bos|>', eos_token='<|eos|>', pad_token='<|eos|>', padding_side='left', model_max_length=sequence_len, add_bos_token=True, additional_special_tokens=[s for s, _ in specials if s != '<|bos|>'])
    tokenizer.save_pretrained(output_dir)
    return enc.n_vocab


def convert(checkpoint_dir, tokenizer_dir, output_dir, step=None, results_markdown=None, assets_dir=None, tokenizer_sha256=None, overwrite=False):
    if (results_markdown is None) != (assets_dir is None):
        raise ValueError('Provide both results_markdown and assets_dir, or neither')
    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    output_resolved = output_dir.resolve()
    inputs = [checkpoint_dir, Path(tokenizer_dir)]
    if results_markdown:
        inputs.extend([Path(results_markdown), Path(assets_dir)])
    if any(path.resolve().is_relative_to(output_resolved) for path in inputs):
        raise ValueError('Output directory must not contain any input')

    def check_output():
        if output_dir.is_symlink():
            raise ValueError('Output directory must not be a symlink')
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f'Output already exists: {output_dir}; use --overwrite')
            marker = output_dir / 'config.json'
            try:
                is_export = json.loads(marker.read_text()).get('model_type') == 'rh_nanochat'
            except (OSError, ValueError, AttributeError):
                is_export = False
            if not is_export:
                raise ValueError('Refusing to overwrite a directory that is not an RH nanochat export')
    check_output()
    if step is None:
        steps = [int(m.group(1)) for p in checkpoint_dir.glob('model_*.pt') if (m := re.fullmatch(r'model_(\d+)\.pt', p.name))]
        if not steps:
            raise FileNotFoundError(f'No numbered checkpoints in {checkpoint_dir}')
        step = max(steps)
    meta = json.loads((checkpoint_dir / f'meta_{step:06d}.json').read_text())
    c = meta['model_config']
    actual_hash = hashlib.sha256((Path(tokenizer_dir) / 'tokenizer.pkl').read_bytes()).hexdigest()
    recorded_hash = meta.get('tokenizer_sha256') or meta.get('provenance', {}).get('tokenizer_sha256')
    if not recorded_hash and not tokenizer_sha256:
        raise ValueError('Checkpoint has no tokenizer_sha256; supply the verified training tokenizer hash with --tokenizer-sha256')
    if recorded_hash and tokenizer_sha256 and recorded_hash.lower() != tokenizer_sha256.lower():
        raise ValueError('Supplied tokenizer hash conflicts with checkpoint provenance')
    if actual_hash != (recorded_hash or tokenizer_sha256).lower():
        raise ValueError('Tokenizer SHA256 does not match the checkpoint training tokenizer')
    results = Path(results_markdown).read_text(encoding='utf-8') if results_markdown else None
    assets = []
    if results is not None:
        image_pattern = r'!\[[^\]]*\]\((assets/[^\s)]+)(?:\s+[^)]*)?\)'
        if re.search(r'(?:\./)?assets/', re.sub(image_pattern, '', results)):
            raise ValueError('Asset references must use inline ![alt](assets/image.png) syntax')
        for link in re.findall(image_pattern, results):
            relative = Path(link).relative_to('assets')
            source = Path(assets_dir) / relative
            if relative.is_absolute() or '..' in relative.parts or any((Path(assets_dir) / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)) or not source.is_file() or source.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.svg'}:
                raise ValueError(f'Invalid or missing image: {link}')
            if not source.resolve().is_relative_to(Path(assets_dir).resolve()):
                raise ValueError(f'Image outside asset directory: {link}')
            assets.append((source, relative))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.rh-export-') as temp:
        dest = Path(temp) / 'model'
        dest.mkdir()
        vocab = convert_tokenizer(tokenizer_dir, dest, c['sequence_len'])
        if vocab != c['vocab_size']:
            raise ValueError(f'Tokenizer vocab {vocab} != model vocab {c["vocab_size"]}')
        with open(Path(tokenizer_dir) / 'tokenizer.pkl', 'rb') as f:
            special_tokens = pickle.load(f)._special_tokens
            bos, eos = special_tokens['<|bos|>'], special_tokens['<|eos|>']
        config = RHNanoChatConfig(vocab_size=c['vocab_size'], hidden_size=c['n_embd'], intermediate_size=4*c['n_embd'], num_hidden_layers=c['n_layer'], num_attention_heads=c['n_head'], num_key_value_heads=c['n_kv_head'], max_position_embeddings=c['sequence_len'], window_pattern=c.get('window_pattern', 'L'), bos_token_id=bos, eos_token_id=eos, pad_token_id=eos, tie_word_embeddings=False, tokenizer_sha256=actual_hash, checkpoint_step=step)
        config.auto_map = {'AutoConfig': 'configuration_rh_nanochat.RHNanoChatConfig', 'AutoModelForCausalLM': 'modeling_rh_nanochat.RHNanoChatForCausalLM'}
        model = RHNanoChatForCausalLM(config)
        state = torch.load(checkpoint_dir / f'model_{step:06d}.pt', map_location='cpu', weights_only=True)
        state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        state.setdefault('resid_lambdas', torch.ones(c['n_layer']))
        state.setdefault('x0_lambdas', torch.zeros(c['n_layer']))
        model.load_state_dict(state, strict=True, assign=True)
        model.save_pretrained(dest, safe_serialization=True)
        for name in ('configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
            shutil.copyfile(Path(__file__).parent / name, dest / name)
        run = meta.get('user_config', {}).get('run', 'unknown')
        card = f'''---\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- lean\n- riemann-nano\n---\n# riemann-nano\n\nTrained on Alpöge's [Lean 4 formalization](https://alpo.ge/) as part of a data-constrained scaling study informed by [Muennighoff et al.](https://arxiv.org/abs/2305.16264). Record the data license, training split and limitations before sharing.\n\nRun: `{run}`; step: {step}; validation BPB: {meta.get('val_bpb', 'unknown')}. Configuration: {json.dumps(c, sort_keys=True)}. Training tokenizer SHA256: `{actual_hash}`.\n\nRequires transformers 5.24. Load both the model and tokenizer with `trust_remote_code=True`.\n'''
        if results is not None:
            card += '\n' + results.strip() + '\n'
            for source, relative in assets:
                target = dest / 'assets' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        else:
            card += '\n## Scaling-law results\n\nAdd isoFLOP curves, best loss, optimal N and D with fitted exponents, tokens:params, per-directory difficulty and the optimal-configurations table before sharing the final model.\n'
        (dest / 'README.md').write_text(card, encoding='utf-8')
        # Preserve the old output until the complete replacement is built.
        check_output()
        # The backup is outside the temporary directory so a failed restore is recoverable.
        old = Path(temp + '-backup')
        if output_dir.exists():
            output_dir.rename(old)
        try:
            dest.rename(output_dir)
        except Exception:
            if old.exists():
                try:
                    old.rename(output_dir)
                except OSError as exc:
                    raise RuntimeError(f'Could not restore previous export; backup remains at {old}') from exc
            raise
        if old.exists():
            shutil.rmtree(old)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint_dir'); p.add_argument('tokenizer_dir'); p.add_argument('output_dir')
    p.add_argument('--step', type=int)
    p.add_argument('--tokenizer-sha256', help='Verified training tokenizer hash for legacy checkpoints')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--results-markdown'); p.add_argument('--assets-dir')
    args = p.parse_args()
    convert(args.checkpoint_dir, args.tokenizer_dir, args.output_dir, args.step, args.results_markdown, args.assets_dir, args.tokenizer_sha256, args.overwrite)

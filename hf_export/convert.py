"""Convert a nanochat checkpoint: python -m hf_export.convert CHECKPOINT_DIR TOKENIZER_DIR OUTPUT_DIR."""
import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import torch
from tiktoken.load import dump_tiktoken_bpe
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter
from .configuration_rh_nanochat import RHNanoChatConfig
from .modeling_rh_nanochat import RHNanoChatForCausalLM


def convert_tokenizer(tokenizer_dir, output_dir):
    # Only load pickle files from trusted sources: pickle can execute arbitrary code.
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
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<|bos|>', additional_special_tokens=[s for s, _ in specials if s != '<|bos|>'])
    tokenizer.save_pretrained(output_dir)
    return enc.n_vocab


def convert(checkpoint_dir, tokenizer_dir, output_dir, step=None, results_markdown=None, assets_dir=None):
    if (results_markdown is None) != (assets_dir is None):
        raise ValueError("Provide both results_markdown and assets_dir, or neither")
    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    if step is None:
        step = max(int(p.stem.split('_')[-1]) for p in checkpoint_dir.glob('model_*.pt'))
    with open(checkpoint_dir / f'meta_{step:06d}.json') as f:
        meta = json.load(f)
    c = meta['model_config']
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = convert_tokenizer(tokenizer_dir, output_dir)
    if vocab != c['vocab_size']:
        raise ValueError(f'Tokenizer vocab {vocab} != model vocab {c["vocab_size"]}')
    config = RHNanoChatConfig(vocab_size=c['vocab_size'], hidden_size=c['n_embd'], intermediate_size=4*c['n_embd'], num_hidden_layers=c['n_layer'], num_attention_heads=c['n_head'], num_key_value_heads=c['n_kv_head'], max_position_embeddings=c['sequence_len'], window_pattern=c.get('window_pattern', 'L'), bos_token_id=None, eos_token_id=None, pad_token_id=None, tie_word_embeddings=False)
    config.auto_map = {'AutoConfig': 'configuration_rh_nanochat.RHNanoChatConfig', 'AutoModelForCausalLM': 'modeling_rh_nanochat.RHNanoChatForCausalLM'}
    model = RHNanoChatForCausalLM(config)
    state = torch.load(checkpoint_dir / f'model_{step:06d}.pt', map_location='cpu', weights_only=True)
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    state.setdefault('resid_lambdas', torch.ones(c['n_layer']))
    state.setdefault('x0_lambdas', torch.zeros(c['n_layer']))
    model.load_state_dict(state, strict=True, assign=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    for name in ('configuration_rh_nanochat.py', 'modeling_rh_nanochat.py'):
        (output_dir / name).write_text((Path(__file__).parent / name).read_text())
    card = '''---\nlibrary_name: transformers\n---\n# riemann-nano\n\nThis model was trained on the Lean 4 formalization of the Riemann Hypothesis proof by Alpöge (https://alpo.ge/). Complete this card with the exact data split, training run, license, evaluation and limitations before sharing.\n\nThe model is part of a data-constrained scaling-law study, informed by Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264).\n\nRequires transformers 5.24 and `trust_remote_code=True`.\n'''
    if results_markdown is not None:
        card += '\n' + Path(results_markdown).read_text(encoding='utf-8').strip() + '\n'
        shutil.copytree(assets_dir, output_dir / 'assets', dirs_exist_ok=True)
    else:
        card += ('\n## Scaling-law results\n\n'
                 'Add the sweep results and plots before publishing the final model: '
                 'isoFLOP curves, best loss against compute, optimal N and D with fitted exponents, '
                 'tokens:params against compute, the per-directory difficulty map, and the '
                 'optimal-configurations table. Embed plots using relative `assets/` links.\n')
    (output_dir / 'README.md').write_text(card, encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint_dir')
    p.add_argument('tokenizer_dir')
    p.add_argument('output_dir')
    p.add_argument('--step', type=int)
    p.add_argument('--results-markdown', help='Sweep results section containing relative assets/ links')
    p.add_argument('--assets-dir', help='Directory of plot assets to copy into assets/')
    args = p.parse_args()
    convert(args.checkpoint_dir, args.tokenizer_dir, args.output_dir, args.step, args.results_markdown, args.assets_dir)

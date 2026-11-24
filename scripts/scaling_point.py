"""Calculate a sweep point using exactly the architecture and FLOP estimate used by base_train."""
import argparse
import json
import torch
from nanochat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--flops', type=float, required=True)
parser.add_argument('--vocab-size', type=int, required=True)
parser.add_argument('--head-dim', type=int, default=128)
args = parser.parse_args()
width = ((args.depth * 64 + args.head_dim - 1) // args.head_dim) * args.head_dim
heads = width // args.head_dim
with torch.device('meta'):
    model = GPT(GPTConfig(sequence_len=2048, vocab_size=args.vocab_size,
                          n_layer=args.depth, n_head=heads, n_kv_head=heads,
                          n_embd=width, window_pattern='SSSL'))
counts = model.num_scaling_params()
params = counts['transformer_matrices'] + counts['lm_head']
flops_per_token = model.estimate_flops()
print(json.dumps({'effective_params': params, 'flops_per_token': flops_per_token,
                  'tokens_per_param': args.flops / flops_per_token / params}))

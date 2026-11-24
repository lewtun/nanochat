"""Check a saved checkpoint's full matched-validation BPB across processes."""
import argparse
import hashlib
import json
from pathlib import Path

from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import compute_init, compute_cleanup
from nanochat.loss_eval import evaluate_bpb, full_document_validation_batches
from nanochat.tokenizer import get_token_bytes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
_, rank, _, world, device = compute_init(args.device)
manifest = json.loads(args.manifest.read_text())
sha = hashlib.sha256((args.tokenizer / 'tokenizer.pkl').read_bytes()).hexdigest()
if sha != manifest['tokenizer_sha256']:
    raise ValueError('Tokenizer does not match manifest')
model, tokenizer, metadata = build_model(str(args.checkpoint), find_last_step(str(args.checkpoint)), device, 'eval')
if sha != metadata['tokenizer_sha256'] or metadata['tokenizer_bucket_path'] != manifest['tokenizer_bucket_path']:
    raise ValueError('Tokenizer does not match checkpoint')
batches = full_document_validation_batches(tokenizer, args.data_dir / manifest['validation_shard'],
                                           metadata['device_batch_size'], model.config.sequence_len,
                                           rank, world, device)
bpb = evaluate_bpb(model, batches, None, get_token_bytes(device))
if rank == 0:
    result = {'checkpoint': args.checkpoint.name, 'world_size': world, 'validation_bpb': bpb,
              'checkpoint_validation_bpb': metadata['val_bpb'], 'tokenizer_sha256': sha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)
compute_cleanup()

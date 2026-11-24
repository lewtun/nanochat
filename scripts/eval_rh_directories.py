"""Evaluate held-out Lean documents by source area, using one BOS per document.

Checkpoint paths must be local (sync the kept bucket checkpoints first). Each
segment starts a new context; every text token is scored once, with no padding
included in the loss or byte denominator.
"""
import argparse
import hashlib
import os
from collections import defaultdict
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import torch
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.common import get_base_dir
from scripts.prepare_rh_data import area

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--unseen-data-dir', type=Path, help='unseen-area parquet directory (defaults to sibling unseen_data)')
parser.add_argument('--difficulty-data-dir', type=Path, help='difficulty-only parquet directory (defaults to sibling difficulty_data)')
parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--device', default='cuda')
parser.add_argument('--force', action='store_true', help='replace existing area results')
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
unseen_dir = args.unseen_data_dir or args.data_dir.parent / manifest['unseen_data_prefix']
difficulty_dir = args.difficulty_data_dir or args.data_dir.parent / manifest['difficulty_data_prefix']
sources = [('matched', manifest['validation_files'], args.data_dir / manifest['validation_shard']),
           ('difficulty_only', manifest['difficulty_files'], difficulty_dir / manifest['difficulty_shard']),
           ('unseen', manifest['unseen_files'], unseen_dir / manifest['unseen_shard'])]
for label, paths, shard in sources:
    if pq.ParquetFile(shard).metadata.num_rows != len(paths):
        raise ValueError(f'{label} shard and manifest document counts differ')
args.output_dir.mkdir(parents=True, exist_ok=True)
for checkpoint in args.checkpoints:
    destination = args.output_dir / f'{checkpoint.name}.json'
    if destination.exists() and not args.force:
        print(f'Skipping existing {destination}', flush=True)
        continue
    print(f'Evaluating {checkpoint}', flush=True)
    step = find_last_step(str(checkpoint))
    device = torch.device(args.device)
    model, tokenizer, meta = build_model(str(checkpoint), step, device, 'eval')
    tokenizer_dir = Path(os.environ.get('NANOCHAT_TOKENIZER_DIR', Path(get_base_dir()) / 'tokenizer'))
    actual_sha = hashlib.sha256((tokenizer_dir / 'tokenizer.pkl').read_bytes()).hexdigest()
    if actual_sha != manifest.get('tokenizer_sha256') or actual_sha != meta.get('tokenizer_sha256'):
        raise ValueError(f'{checkpoint}: tokenizer hash does not match both manifest and checkpoint')
    if meta.get('tokenizer_bucket_path') != manifest.get('tokenizer_bucket_path'):
        raise ValueError(f'{checkpoint}: tokenizer bucket path does not match manifest')
    if not manifest.get('document_eos') or not meta.get('user_config', {}).get('document_eos'):
        raise ValueError(f'{checkpoint}: expected the canonical document-EOS setup')
    eos = tokenizer.encode_special('<|eos|>')
    seq_len = model.config.sequence_len
    stats = defaultdict(lambda: {'nats': 0.0, 'bytes': 0, 'tokens': 0, 'documents': 0})
    with torch.inference_mode():
        for evaluation_set, paths, shard in sources:
            index = 0
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=128, columns=['text']):
                for text in batch.column(0).to_pylist():
                    path = paths[index]
                    index += 1
                    row = stats[area(Path(path))]
                    if row.get('evaluation_set', evaluation_set) != evaluation_set:
                        raise ValueError('An area occurs in both matched and unseen sets')
                    row['evaluation_set'] = evaluation_set
                    text_ids = tokenizer.encode(text)
                    ids = [tokenizer.get_bos_token_id()] + text_ids + [eos]
                    row['bytes'] += len(text.encode('utf-8'))
                    row['tokens'] += len(text_ids)
                    row['documents'] += 1
                    for start in range(0, len(ids)-1, seq_len):
                        segment = ids[start:start+seq_len+1]
                        inputs, targets = segment[:-1], [(-1 if token == eos else token) for token in segment[1:]]
                        if len(inputs) == 1:
                            inputs.append(tokenizer.get_bos_token_id())
                            targets.append(-1)
                        x = torch.tensor([inputs], dtype=torch.long, device=device)
                        y = torch.tensor([targets], dtype=torch.long, device=device)
                        row['nats'] += model(x, y, loss_reduction='sum').item()
    for row in stats.values():
        row['bits_per_byte'] = row['nats'] / (math.log(2) * row['bytes']) if row['bytes'] else None
    payload = {'checkpoint': checkpoint.name, 'step': step, 'sequence_length': seq_len,
               'method': 'document BOS/text/EOS, nonoverlapping context windows; EOS excluded from BPB',
               'tokenizer_sha256': actual_sha, 'tokenizer_bucket_path': meta['tokenizer_bucket_path'],
               'unseen_areas': manifest['unseen_areas'], 'area_evaluation_set': manifest['area_evaluation_set'], 'matched_unrepresented_areas': manifest['matched_unrepresented_areas'],
               'uncovered_areas': manifest['uncovered_areas'], 'file_level_areas': manifest.get('file_level_areas', {}), 'areas': dict(sorted(stats.items()))}
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(destination)

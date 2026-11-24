"""Prepare a deterministic, directory-held-out Lean corpus for nanochat.

Run with --source .../lean/src --output .../runtime, then train the tokenizer:
NANOCHAT_BASE_DIR=.../runtime NANOCHAT_DATA_DIR=.../runtime/base_data \
  python -m scripts.tok_train --vocab-size 16384 --document-eos
Finally run this script with --count-tokens --bucket namespace/name to add
counts, fingerprints and canonical bucket paths to the manifest.
"""
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import random
import os
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def area(path):
    parts = path.parent.parts
    return '/'.join(parts[:3 if parts and parts[0] == 'Development' else 2]) or '(root)'


def atomic_manifest(path, payload):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.manifest.', delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def prepare(source, output, seed, shard_size, source_tarball=None):
    manifest_path = output / 'data_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not old.get('document_eos') or old.get('asset_prefix') != 'rh' or old.get('data_prefix') != 'base_data':
            raise ValueError('Existing data is not the canonical EOS layout; use a fresh output directory')
        if not all(key in old for key in ('file_level_areas', 'area_evaluation_set', 'difficulty_shard', 'unseen_shard')):
            raise ValueError('Existing data uses an older split; use a fresh output directory')
        if old.get('seed') != seed or old.get('shard_size') != shard_size:
            raise ValueError('Existing manifest has different preparation settings')
        if not all((output / old['data_prefix'] / name).is_file() for name in old['parquet_files']):
            raise FileNotFoundError('Existing manifest has missing shards')
        for split in ('unseen', 'difficulty'):
            if not (output / old[split + '_data_prefix'] / old[split + '_shard']).is_file():
                raise FileNotFoundError(f'Existing manifest has missing {split} shard')
        print(f'Using existing prepared data: {manifest_path}')
        return
    files = sorted(source.rglob('*.lean'))
    if not files:
        raise ValueError('No Lean files found')
    dirs = {f.parent for f in files}
    ancestors = {parent for directory in dirs for parent in directory.parents}
    leaves = dirs - ancestors
    eligible = defaultdict(lambda: defaultdict(int))
    area_bytes = defaultdict(int)
    for f in files:
        relative = f.relative_to(source)
        size = f.stat().st_size
        area_bytes[area(relative)] += size
        if f.parent in leaves:
            eligible[area(relative)][f.parent] += size
    # An entire single-leaf area is kept as a transfer-only evaluation set.
    unseen_areas = {name for name, sizes in eligible.items()
                    if len(sizes) == 1 and sum(sizes.values()) == area_bytes[name]}
    unseen = [f for f in files if area(f.relative_to(source)) in unseen_areas]
    rng = random.Random(seed)
    held = set()
    matched_unrepresented = []
    for name in sorted(set(eligible) - unseen_areas):
        entries = sorted(eligible[name].items(), key=lambda item: str(item[0]))
        target = 0.03 * area_bytes[name]
        best, error = [], target
        # Randomize whole-leaf proposals, keeping the closest 3% subset.
        # Empty is allowed where every eligible leaf is grossly oversized.
        for _ in range(200):
            rng.shuffle(entries)
            selected, size = [], 0
            for directory, n in entries:
                if size + n < area_bytes[name] and abs(size + n - target) < abs(size - target):
                    selected.append(directory)
                    size += n
            if abs(size - target) < error:
                best, error = selected[:], abs(size - target)
        held.update(best)
        if not best:
            matched_unrepresented.append(name)
    difficulty_dirs = set()
    for name in matched_unrepresented:
        options = [(size, str(directory), directory) for directory, size in eligible[name].items() if size < area_bytes[name]]
        if not options:
            raise ValueError(f'No held-out leaf can leave training data in {name}')
        difficulty_dirs.add(min(options)[2])
    fallback_areas = sorted(set(area_bytes) - set(eligible))
    file_holdouts = {}
    for name in fallback_areas:
        candidates = [f for f in files if area(f.relative_to(source)) == name]
        count = min(len(candidates) - 1, max(1, round(len(candidates) * 0.03)))
        if count <= 0:
            raise ValueError(f'Cannot leave both train and validation files in {name}')
        file_holdouts[name] = sorted(rng.sample(candidates, count))
    held_files = {f for selected in file_holdouts.values() for f in selected}
    unseen_set = set(unseen)
    difficulty = [f for f in files if f not in unseen_set and f.parent in difficulty_dirs]
    difficulty_set = set(difficulty)
    val = [f for f in files if f not in unseen_set and (f.parent in held or f in held_files)]
    train = [f for f in files if f not in unseen_set and f not in difficulty_set and f.parent not in held and f not in held_files]
    validation_hashes = {hashlib.sha256(f.read_bytes()).digest() for f in val + unseen + difficulty}
    duplicates = [f for f in train if hashlib.sha256(f.read_bytes()).digest() in validation_hashes]
    duplicate_set = set(duplicates)
    train = [f for f in train if f not in duplicate_set]
    random.Random(seed).shuffle(train)
    data = output / 'base_data'
    data.mkdir(parents=True, exist_ok=True)
    if any(data.glob('*.parquet')):
        raise FileExistsError(f'{data} already contains shards; use a fresh output directory')
    names = []
    for i in range(0, len(train), shard_size):
        name = f'shard_{i // shard_size:05d}.parquet'
        write(data / name, train[i:i + shard_size])
        names.append(name)
    name = 'shard_99999.parquet'
    write(data / name, val)
    names.append(name)
    unseen_data = output / 'unseen_data'
    unseen_data.mkdir(parents=True, exist_ok=True)
    write(unseen_data / 'unseen.parquet', unseen)
    difficulty_data = output / 'difficulty_data'
    difficulty_data.mkdir(parents=True, exist_ok=True)
    write(difficulty_data / 'difficulty.parquet', difficulty)
    def stats(items):
        return {'files': len(items), 'bytes': sum(f.stat().st_size for f in items)}
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()
    source_sha = None
    if source_tarball:
        with source_tarball.open('rb') as handle:
            source_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    per_area = {name: {'train_bytes': 0, 'validation_bytes': 0, 'unseen_bytes': 0, 'difficulty_bytes': 0} for name in area_bytes}
    for group, items in [('train', train), ('validation', val), ('unseen', unseen), ('difficulty', difficulty)]:
        for f in items:
            per_area[area(f.relative_to(source))][group + '_bytes'] += f.stat().st_size
    train_bytes, val_bytes = stats(train)['bytes'], stats(val)['bytes']
    if not train_bytes or not val_bytes:
        raise ValueError('Matched split requires nonempty training and validation data')
    tv = 0.5 * sum(abs(v['train_bytes']/train_bytes - v['validation_bytes']/val_bytes) for v in per_area.values())
    manifest = {'schema_version': 1, 'status': 'PREPARED', 'dataset': 'alpo.ge/lean',
                'git_sha': git_sha, 'source_tarball_sha256': source_sha,
                'excluded_duplicate_train_files': [str(f.relative_to(source)) for f in duplicates],
                'seed': seed, 'shard_size': shard_size, 'asset_prefix': 'rh', 'data_prefix': 'base_data', 'document_eos': True, 'parquet_files': names,
                'num_train_shards': len(names)-1, 'validation_shard': name,
                'held_out_directories': sorted(str(d.relative_to(source)) for d in held),
                'validation_files': [str(f.relative_to(source)) for f in val],
                'unseen_files': [str(f.relative_to(source)) for f in unseen],
                'unseen_shard': 'unseen.parquet', 'unseen_data_prefix': 'unseen_data',
                'difficulty_shard': 'difficulty.parquet', 'difficulty_data_prefix': 'difficulty_data',
                'difficulty_files': [str(f.relative_to(source)) for f in difficulty],
                'difficulty_held_out_directories': sorted(str(d.relative_to(source)) for d in difficulty_dirs),
                'unseen_areas': sorted(unseen_areas),
                'matched_unrepresented_areas': matched_unrepresented,
                'area_split_bytes': dict(sorted(per_area.items())),
                'area_evaluation_set': {name: ('unseen' if name in unseen_areas else 'difficulty_only' if name in matched_unrepresented else 'matched') for name in sorted(area_bytes)},
                'train_validation_area_tv': tv,
                'file_level_areas': {name: [str(f.relative_to(source)) for f in selected] for name, selected in file_holdouts.items()},
                'uncovered_areas': [],
                'train': stats(train), 'validation': stats(val), 'unseen': stats(unseen), 'difficulty': stats(difficulty),
                'area_bytes': dict(sorted(area_bytes.items()))}
    atomic_manifest(output / 'data_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('train', 'validation', 'uncovered_areas')}, indent=2))


def write(path, files):
    writer = pq.ParquetWriter(path, pa.schema([('text', pa.string())]), compression='zstd')
    try:
        for start in range(0, len(files), 256):
            texts = [f.read_text(encoding='utf-8') for f in files[start:start+256]]
            writer.write_table(pa.table({'text': texts}))
    finally:
        writer.close()


def count(output, bucket):
    from nanochat.tokenizer import RustBPETokenizer
    tokenizer = RustBPETokenizer.from_directory(output / 'tokenizer')
    path = output / 'data_manifest.json'
    manifest = json.loads(path.read_text())
    for split, names in [('train', manifest['parquet_files'][:-1]), ('validation', manifest['parquet_files'][-1:]), ('unseen', [manifest['unseen_shard']]), ('difficulty', [manifest['difficulty_shard']])]:
        tokens = 0
        for name in names:
            directory = manifest[split + '_data_prefix'] if split in ('unseen', 'difficulty') else manifest['data_prefix']
            parquet = pq.ParquetFile(output / directory / name)
            for batch in parquet.iter_batches(batch_size=256, columns=['text']):
                tokens += sum(len(ids) for ids in tokenizer.encode(batch.column(0).to_pylist()))
            print(f'{split} {name}: {tokens:,} tokens', flush=True)
        manifest[split]['tokens'] = tokens
        if manifest.get('document_eos'):
            manifest[split]['tokens_with_boundaries'] = tokens + 2 * manifest[split]['files']
        manifest[split]['bytes_per_token'] = manifest[split]['bytes'] / tokens
    if manifest.get('document_eos') and '<|eos|>' not in tokenizer.get_special_tokens():
        raise ValueError('Document-EOS manifest requires an EOS tokenizer')
    manifest['tokenizer_special_tokens'] = sorted(tokenizer.get_special_tokens())
    if manifest.get('document_eos'):
        manifest['eos_token_id'] = tokenizer.encode_special('<|eos|>')
    manifest['tokenizer_sha256'] = hashlib.sha256((output / 'tokenizer' / 'tokenizer.pkl').read_bytes()).hexdigest()
    manifest['tokenizer_relative_path'] = 'tokenizer/tokenizer.pkl'
    manifest['tokenizer_bucket_path'] = f'hf://buckets/{bucket}/rh/tokenizer/tokenizer.pkl'
    manifest['data_bucket_path'] = f'hf://buckets/{bucket}/rh/base_data'
    manifest['unseen_bucket_path'] = f'hf://buckets/{bucket}/rh/unseen_data'
    manifest['difficulty_bucket_path'] = f'hf://buckets/{bucket}/rh/difficulty_data'
    manifest['tokenizer_vocab_size'] = tokenizer.get_vocab_size()
    manifest['status'] = 'COMPLETE'
    atomic_manifest(path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--source-tarball', type=Path, help='original archive for source fingerprint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--count-tokens', action='store_true')
    parser.add_argument('--bucket', help='namespace/bucket for canonical manifest paths')
    args = parser.parse_args()
    if args.count_tokens:
        if not args.bucket or args.bucket.count('/') != 1 or any(c.isspace() for c in args.bucket):
            parser.error('--bucket namespace/name is required with --count-tokens')
        count(args.output, args.bucket)
    else:
        if args.source is None or args.shard_size <= 0:
            parser.error('--source and positive --shard-size required')
        prepare(args.source, args.output, args.seed, args.shard_size, args.source_tarball)

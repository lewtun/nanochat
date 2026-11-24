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
        if 'file_level_areas' not in old:
            raise ValueError('Existing data uses the older split; use a fresh output directory')
        if old.get('seed') != seed or old.get('shard_size') != shard_size:
            raise ValueError('Existing manifest has different preparation settings')
        if not all((output / old['data_prefix'] / name).is_file() for name in old['parquet_files']):
            raise FileNotFoundError('Existing manifest has missing shards')
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
    # The minimum feasible coverage minimizes the deviation from the 3% target.
    held = {min(dirsizes, key=lambda d: (dirsizes[d], str(d))) for dirsizes in eligible.values()}
    fallback_areas = sorted(set(area_bytes) - set(eligible))
    file_holdouts = {}
    rng = random.Random(seed)
    for name in fallback_areas:
        candidates = [f for f in files if area(f.relative_to(source)) == name]
        count = max(1, round(len(candidates) * 0.03))
        file_holdouts[name] = sorted(rng.sample(candidates, count))
    held_files = {f for selected in file_holdouts.values() for f in selected}
    train = [f for f in files if f.parent not in held and f not in held_files]
    val = [f for f in files if f.parent in held or f in held_files]
    validation_hashes = {hashlib.sha256(f.read_bytes()).digest() for f in val}
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
    def stats(items):
        return {'files': len(items), 'bytes': sum(f.stat().st_size for f in items)}
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()
    source_sha = hashlib.file_digest(source_tarball.open('rb'), 'sha256').hexdigest() if source_tarball else None
    manifest = {'schema_version': 1, 'status': 'PREPARED', 'dataset': 'alpo.ge/lean',
                'git_sha': git_sha, 'source_tarball_sha256': source_sha,
                'excluded_duplicate_train_files': [str(f.relative_to(source)) for f in duplicates],
                'seed': seed, 'shard_size': shard_size, 'asset_prefix': 'rh', 'data_prefix': 'base_data', 'document_eos': True, 'parquet_files': names,
                'num_train_shards': len(names)-1, 'validation_shard': name,
                'held_out_directories': sorted(str(d.relative_to(source)) for d in held),
                'validation_files': [str(f.relative_to(source)) for f in val],
                'file_level_areas': {name: [str(f.relative_to(source)) for f in selected] for name, selected in file_holdouts.items()},
                'uncovered_areas': [],
                'train': stats(train), 'validation': stats(val),
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
    for split, names in [('train', manifest['parquet_files'][:-1]), ('validation', manifest['parquet_files'][-1:])]:
        tokens = 0
        for name in names:
            parquet = pq.ParquetFile(output / manifest['data_prefix'] / name)
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

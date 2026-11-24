"""Annotate completed bucket runs with the tokenizer they actually used.

Each run's immutable asset prefix in provenance determines the manifest and
bucket tokenizer. Only COMPLETE runs are touched, and an existing conflicting
hash or path is an error. Safe to rerun. Uses the authenticated `hf` CLI.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bucket', required=True, help='namespace/bucket')
parser.add_argument('--labels', nargs='+', required=True)
parser.add_argument('--apply', action='store_true', help='write annotations to the bucket')
args = parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', args.bucket):
    parser.error('invalid bucket')
if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', label) for label in args.labels):
    parser.error('invalid run label')
base = f'hf://buckets/{args.bucket}'


def hf(*args):
    return subprocess.run(['hf', *args], check=True, capture_output=True, text=True).stdout


def annotate(obj, sha, uri):
    for key, value in [('tokenizer_sha256', sha), ('tokenizer_bucket_path', uri)]:
        if obj.get(key) not in (None, '', value):
            raise ValueError(f'Conflicting {key}: {obj[key]!r} != {value!r}')
        obj[key] = value
    return obj


with tempfile.TemporaryDirectory(prefix='rh-tokenizer-provenance-') as tmp:
    root = Path(tmp)
    for label in args.labels:
        prefix = f'scaling_laws/{label}'
        status = root / 'complete.json'
        provenance = root / 'provenance.json'
        hf('buckets', 'cp', f'{base}/{prefix}/COMPLETE', str(status))
        if json.loads(status.read_text())['state'] != 'COMPLETE':
            raise ValueError(f'{label} is not complete')
        hf('buckets', 'cp', f'{base}/{prefix}/provenance.json', str(provenance))
        info = json.loads(provenance.read_text())
        if info['run_label'] != label or info['data_bucket'] != args.bucket:
            raise ValueError(f'{label}: provenance does not match the requested run')
        asset = info.get('asset_prefix', '')
        if asset and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', asset):
            raise ValueError(f'{label}: invalid recorded asset prefix')
        if label.startswith('rh-v2-') and asset != 'rh_v2':
            raise ValueError(f'{label}: missing revised asset prefix')
        asset_root = f'{asset}/' if asset else ''
        uri = f'{base}/{asset_root}tokenizer/tokenizer.pkl'
        tokenizer = root / 'tokenizer.pkl'
        manifest = root / 'manifest.json'
        hf('buckets', 'cp', uri, str(tokenizer))
        hf('buckets', 'cp', f'{base}/{asset_root}data_manifest.json', str(manifest))
        sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        if json.loads(manifest.read_text()).get('tokenizer_sha256') != sha:
            raise ValueError(f'{label}: tokenizer hash does not match its manifest')
        paths = json.loads(hf('buckets', 'ls', f'{base}/{prefix}/checkpoints', '--recursive', '--format', 'json'))
        metas = sorted(x['path'] for x in paths if re.search(r'/meta_\d+\.json$', x.get('path', '')))
        if not metas:
            raise ValueError(f'{label}: no checkpoint metadata found')
        annotate(info, sha, uri)
        provenance.write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
        if args.apply:
            hf('buckets', 'cp', str(provenance), f'{base}/{prefix}/provenance.json')
        for path in metas:
            meta = root / 'meta.json'
            hf('buckets', 'cp', f'{base}/{path}', str(meta))
            data = annotate(json.loads(meta.read_text()), sha, uri)
            meta.write_text(json.dumps(data, indent=2) + '\n')
            if args.apply:
                hf('buckets', 'cp', str(meta), f'{base}/{path}')
        print(f'{label}: {len(metas)} checkpoints; sha256={sha}; path={uri}; ' + ('updated' if args.apply else 'dry run'), flush=True)

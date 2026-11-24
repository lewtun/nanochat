import csv
import json
import os
import subprocess
import sys


def test_synthetic_analysis_reports_valid_and_unbracketed_budgets(tmp_path):
    manifest = {'document_eos': True, 'asset_prefix': 'rh', 'tokenizer_sha256': 'abc', 'tokenizer_bucket_path': 'bucket/tokenizer'}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    fields = ['asset_prefix', 'head_dim', 'document_eos', 'tokenizer_sha256', 'tokenizer_bucket_path', 'batch_lr_scale', 'weight_decay_scaled', 'validation_mode', 'model_dim', 'depth', 'num_iterations', 'flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb']
    path = tmp_path / 'rows.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for budget_index, budget in enumerate([1e15, 1e16, 1e17, 1e18]):
            for depth in range(1, 8):
                n = 1e6 * depth
                loss = (depth - (4 if budget_index < 3 else 1)) ** 2 * .03 + .5
                writer.writerow(dict(asset_prefix='rh', head_dim=64, document_eos=1, tokenizer_sha256='abc', tokenizer_bucket_path='bucket/tokenizer', batch_lr_scale=.3, weight_decay_scaled=.2, validation_mode='full_documents', model_dim=64*depth, depth=depth, num_iterations=500, flops_budget=budget, effective_params=n, tokens_trained=1e7*(budget_index+1)/depth, text_tokens_trained=1e7, unique_tokens=1e7, passes=.1, repeat_factor=1, val_bpb=loss))
    output = tmp_path / 'output'
    env = os.environ.copy()
    subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(output)], check=True, env=env, timeout=60)
    status = json.loads((output / 'budget_status.json').read_text())
    assert [v['status'] for v in status.values()].count('valid') == 3
    assert [v['status'] for v in status.values()].count('invalid') == 1
    fits = json.loads((output / 'fits.json').read_text())
    assert fits['fresh']['n'] == 3
    assert fits['data_constrained']['status'] != 'fit'

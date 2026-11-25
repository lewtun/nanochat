import csv
import json
import os
import subprocess
import sys
import pytest


def test_a100_comparison_is_rejected(tmp_path):
    manifest = {'document_eos': True, 'asset_prefix': 'rh', 'tokenizer_sha256': 'abc', 'tokenizer_bucket_path': 'bucket/tokenizer'}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    row = dict(run_name='scaling_rh-c-a100x4-1e19-d33_1e19_d33', asset_prefix='rh', head_dim=64, document_eos=1, tokenizer_sha256='abc', tokenizer_bucket_path='bucket/tokenizer', batch_lr_scale=.3, weight_decay_scaled=.2, validation_mode='full_documents', model_dim=2112, depth=33, num_iterations=500, flops_budget=1e19)
    path = tmp_path / 'a100.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    result = subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(tmp_path / 'out')], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'excluded from the H200-only grid' in result.stderr


def test_synthetic_analysis_reports_valid_and_unbracketed_budgets(tmp_path):
    manifest = {'document_eos': True, 'asset_prefix': 'rh', 'tokenizer_sha256': 'abc', 'tokenizer_bucket_path': 'bucket/tokenizer', 'train': {'files': 100, 'tokens': 1000}, 'validation': {'files': 10}, 'train_validation_area_tv': .03}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    fields = ['asset_prefix', 'head_dim', 'document_eos', 'tokenizer_sha256', 'tokenizer_bucket_path', 'batch_lr_scale', 'weight_decay_scaled', 'validation_mode', 'model_dim', 'depth', 'num_iterations', 'nproc_per_node', 'flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb']
    path = tmp_path / 'rows.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for budget_index, budget in enumerate([1e15, 1e16, 1e17, 1e18]):
            for depth in range(1, 8):
                n = 1e6 * depth
                loss = (depth - (4 if budget_index < 3 else 1)) ** 2 * .03 + .5
                writer.writerow(dict(asset_prefix='rh', head_dim=64, document_eos=1, tokenizer_sha256='abc', tokenizer_bucket_path='bucket/tokenizer', batch_lr_scale=.3, weight_decay_scaled=.2, validation_mode='full_documents', model_dim=64*depth, depth=depth, num_iterations=500, nproc_per_node=1 if budget_index == 0 else 8, flops_budget=budget, effective_params=n, tokens_trained=1e7*(budget_index+1)/depth, text_tokens_trained=1e7, unique_tokens=1e7, passes=.1, repeat_factor=1, val_bpb=loss))
    output = tmp_path / 'output'
    env = os.environ.copy()
    subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(output)], check=True, env=env, timeout=60)
    status = json.loads((output / 'budget_status.json').read_text())
    assert [v['status'] for v in status.values()].count('valid') == 3
    assert [v['status'] for v in status.values()].count('invalid') == 1
    fits = json.loads((output / 'fits.json').read_text())
    assert fits['fresh']['n'] == 3
    assert fits['all_budgets']['n'] == 3
    assert 'mix fresh and repeated' in fits['all_budgets']['note']
    assert fits['data_constrained']['status'] != 'fit'
    selection = json.loads((output / 'selected_model.json').read_text())
    assert selection['budget'] == 1e17
    assert selection['depth'] == 4
    assert selection['final_validation_bpb'] == .5

    card = (output / 'model_card_results.md').read_text()
    assert '100 documents and 1,000 text tokens' in card
    assert 'total-variation distance 0.030' in card
    assert '1e+15 FLOPs: h200' in card
    assert '1e+16 FLOPs: h200x8' in card
    duplicate = subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), str(path), '--output-dir', str(output)], capture_output=True, text=True, timeout=60)
    assert duplicate.returncode != 0
    assert 'Duplicate budget/depth' in duplicate.stderr

    # Seed floors can differ by data regime; require at least three budgets in
    # each pool before the analysis reports separate noise estimates.
    with path.open() as f:
        records = list(csv.DictReader(f))
    base = [r for r in records if float(r['flops_budget']) == 1e15]
    for budget in (2e17, 4e17, 8e17):
        for row in base:
            r = row.copy()
            r['flops_budget'] = budget
            r['passes'] = 2
            records.append(r)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(output), '--noise-sigma', '0.2', '--repeated-noise-sigma', '0.1'], check=True, env=env, timeout=60)
    status = json.loads((output / 'budget_status.json').read_text())
    assert status[str(1e15)]['bootstrap_noise_pool'] == 'regime'
    assert status[str(1e15)]['bootstrap_noise_sigma'] == pytest.approx(.2)
    assert status[str(2e17)]['bootstrap_noise_sigma'] == pytest.approx(.1)


def test_recovers_identifiable_data_constrained_law(tmp_path):
    import math
    import pytest
    manifest = {'document_eos': True, 'asset_prefix': 'rh', 'tokenizer_sha256': 'abc', 'tokenizer_bucket_path': 'bucket/tokenizer'}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    fields = ['asset_prefix', 'head_dim', 'document_eos', 'tokenizer_sha256', 'tokenizer_bucket_path', 'batch_lr_scale', 'weight_decay_scaled', 'validation_mode', 'model_dim', 'depth', 'num_iterations', 'flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb']
    path = tmp_path / 'rows.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for compute in (1e15, 1e16, 1e17, 1e18, 1e19):
            for depth in range(1, 13):
                n = 1e6 * 2**(depth-1)
                d = compute / (6*n)
                u = min(100e6, d)
                nm, dm, um = n/1e6, d/1e6, u/1e6
                A, B, alpha, beta, E, rd_star, rn_star = 3., 2., .4, .3, .2, 3., 5.
                un = min(nm, (alpha*A/(beta*B))**(1/alpha) * um**(beta/alpha))
                rn, rd = nm/un-1, dm/um-1
                ne = un * (1+rn_star*(-math.expm1(-rn/rn_star)))
                de = um * (1+rd_star*(-math.expm1(-rd/rd_star)))
                loss = A/ne**alpha+B/de**beta+E
                writer.writerow(dict(asset_prefix='rh', head_dim=64, document_eos=1, tokenizer_sha256='abc', tokenizer_bucket_path='bucket/tokenizer', batch_lr_scale=.3, weight_decay_scaled=.2, validation_mode='full_documents', model_dim=64*depth, depth=depth, num_iterations=500, flops_budget=compute, effective_params=n, tokens_trained=d, text_tokens_trained=d, unique_tokens=u, passes=d/100e6, repeat_factor=d/u, val_bpb=loss))
    output = tmp_path / 'output'
    subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(output)], check=True, timeout=60)
    result = json.loads((output / 'fits.json').read_text())['data_constrained']
    assert result['status'] == 'fit'
    assert result['parameters'] == pytest.approx({'A': 3, 'B': 2, 'alpha': .4, 'beta': .3, 'E': .2, 'R_D_star': 3, 'R_N_star': 5}, rel=1e-4)
    assert result['jacobian_condition'] < 1e8
    assert result['standard_errors'] is not None
    assert result['bound_likelihood_ratios']['R_D_star'] > 3.84
    assert result['bound_likelihood_ratios']['R_N_star'] > 3.84
    # The post-fit profile check must not declare unbounded repetition terms
    # identified when the generating law has effectively no repetition penalty.
    import numpy as np
    rng = np.random.default_rng(17)
    with path.open() as f:
        records = list(csv.DictReader(f))
    for row in records:
        n = float(row['effective_params']) / 1e6
        d = float(row['text_tokens_trained']) / 1e6
        u = float(row['unique_tokens']) / 1e6
        un = min(n, (alpha*A/(beta*B))**(1/alpha) * u**(beta/alpha))
        rn, rd = n/un-1, d/u-1
        ne = un * (1 + 10000 * (-math.expm1(-rn/10000)))
        de = u * (1 + 10000 * (-math.expm1(-rd/10000)))
        row['val_bpb'] = A/ne**alpha + B/de**beta + E + rng.normal(0, .005)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    subprocess.run([sys.executable, '-m', 'scripts.analyze_rh_scaling', '--manifest', str(manifest_path), '--results', str(path), '--output-dir', str(output)], check=True, timeout=60)
    result = json.loads((output / 'fits.json').read_text())['data_constrained']
    assert result['status'] == 'unidentified'

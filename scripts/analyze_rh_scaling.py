"""Analyze RH results. Run: uv run --extra cpu --with scipy --with matplotlib python -m scripts.analyze_rh_scaling --manifest ... --results ... --output-dir ...

The fitted loss follows Muennighoff et al., https://arxiv.org/abs/2305.16264,
with N and D expressed in millions; bpB is the fitted loss. Estimates from
unbracketed minima are rejected; weakly identified fits are reported as such.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from nanochat.scaling import fit_local_isoflop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, required=True, help='canonical data manifest')
parser.add_argument('--results', type=Path, nargs='+', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--area-dir', type=Path, help='directory of per-checkpoint area evaluation JSON files')
parser.add_argument('--asset-prefix', default='rh', help='only analyze runs using this dataset and tokenizer version')
parser.add_argument('--head-dim', type=int, default=64)
parser.add_argument('--headline-max-flops', type=float, default=1e17, help='maximum compute budget included in the headline plots and table')
parser.add_argument('--noise-sigma', type=float, default=0, help='independently measured BPB noise floor (pooled with residual estimate conservatively)')
parser.add_argument('--repeated-noise-sigma', type=float, help='independent repeated-data noise floor; pool residual variance separately by regime')
parser.add_argument('--diagnostics', type=Path, help='JSON with controlled seed and optimizer comparisons for the model card')
args = parser.parse_args()
if not np.isfinite(args.noise_sigma) or args.noise_sigma < 0:
    parser.error('--noise-sigma must be finite and nonnegative')
if args.repeated_noise_sigma is not None and (not np.isfinite(args.repeated_noise_sigma) or args.repeated_noise_sigma < 0):
    parser.error('--repeated-noise-sigma must be finite and nonnegative')
manifest = json.loads(args.manifest.read_text())
if not manifest.get('document_eos') or manifest.get('asset_prefix') != args.asset_prefix:
    raise ValueError('Manifest does not describe the canonical EOS dataset')
expected_sha = manifest['tokenizer_sha256']
expected_path = manifest['tokenizer_bucket_path']
rows = []
seen_points = {}
for path in args.results:
    with path.open() as f:
        for r in csv.DictReader(f):
            if (r.get('asset_prefix') != args.asset_prefix or r.get('head_dim') != str(args.head_dim)
                    or r.get('document_eos') != '1' or r.get('tokenizer_sha256') != expected_sha
                    or r.get('tokenizer_bucket_path') != expected_path):
                raise ValueError(f'{path}: result does not match canonical EOS tokenizer and architecture; do not mix experiments')
            if not r.get('batch_lr_scale') or not r.get('weight_decay_scaled'):
                raise ValueError(f'{path}: obsolete run predates FLOP-horizon optimizer correction')
            if r.get('validation_mode') != 'full_documents':
                raise ValueError(f'{path}: RH analysis requires one full document validation pass')
            if int(r['model_dim']) != int(r['depth']) * 64 or int(r['num_iterations']) < 500:
                raise ValueError(f'{path}: incompatible model width or fewer than 500 optimizer steps')
            point = (float(r['flops_budget']), int(r['depth']))
            name = r.get('run_name') or f'{path}:depth={r["depth"]}'
            if 'rh-c-a100' in name:
                raise ValueError(f'{path}: A100 row {name} is excluded from the H200-only grid')
            if point in seen_points:
                raise ValueError(f'Duplicate budget/depth {point}: {seen_points[point]} and {name}')
            seen_points[point] = name
            row = {k: float(r[k]) for k in ('flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb')}
            row['run_name'] = name
            row['depth'] = int(r['depth'])
            row['params_total'] = int(r['params_total']) if r.get('params_total') else None
            row['nproc_per_node'] = int(r['nproc_per_node']) if r.get('nproc_per_node') else None
            rows.append(row)
if not rows:
    raise SystemExit('No results')
args.output_dir.mkdir(parents=True, exist_ok=True)
optimal = []
bootstrap_optima = {}
budget_status = {}
# Pool local residual variance using the two residual degrees of freedom per
# quadratic. Five residuals alone underestimate the observation noise.
pooled_ss, pooled_df = 0.0, 0
regime_ss = {'fresh': 0.0, 'repeated': 0.0}
regime_count = {'fresh': 0, 'repeated': 0}
for budget in sorted({r['flops_budget'] for r in rows}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r: r['effective_params'])
    try:
        coeffs, _ = fit_local_isoflop([r['effective_params'] for r in group], [r['val_bpb'] for r in group])
    except ValueError:
        continue
    best = int(np.argmin([r['val_bpb'] for r in group]))
    window = group[best-2:best+3]
    residual = np.array([r['val_bpb'] for r in window]) - np.polyval(coeffs, np.log10([r['effective_params'] for r in window]))
    ss = float(residual @ residual)
    pooled_ss += ss
    pooled_df += 2
    x = np.log10([r['effective_params'] for r in group])
    vertex = -coeffs[1] / (2 * coeffs[0])
    estimated_passes = 10 ** np.interp(vertex, x, np.log10([r['passes'] for r in group]))
    regime = 'fresh' if estimated_passes <= 1 else 'repeated'
    regime_ss[regime] += ss
    regime_count[regime] += 1
noise_sigma = max(args.noise_sigma, np.sqrt(pooled_ss / pooled_df) if pooled_df else 0.0)
regime_noise = None
if args.repeated_noise_sigma is not None and all(regime_count[r] >= 3 for r in regime_count):
    regime_noise = {
        'fresh': max(args.noise_sigma, np.sqrt(regime_ss['fresh'] / (2 * regime_count['fresh']))),
        'repeated': max(args.repeated_noise_sigma, np.sqrt(regime_ss['repeated'] / (2 * regime_count['repeated']))),
    }
for budget in sorted({r['flops_budget'] for r in rows}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r:r['effective_params'])
    x = np.log10([r['effective_params'] for r in group]); y = [r['val_bpb'] for r in group]
    try:
        (a,b,c), fit_x = fit_local_isoflop([r['effective_params'] for r in group], y)
    except ValueError as error:
        budget_status[str(budget)] = {'status': 'invalid', 'reason': str(error)}
        continue
    z = -b/(2*a); n = 10**z
    d = float(10 ** np.interp(z,x,np.log10([r['tokens_trained'] for r in group])))
    passes = float(10 ** np.interp(z,x,np.log10([r['passes'] for r in group])))
    repeat_factor = max(1.0, passes)
    budget_noise = regime_noise['fresh' if passes <= 1 else 'repeated'] if regime_noise else noise_sigma
    observed = np.asarray(y, dtype=float)
    best_index = int(np.argmin(observed))
    residuals = observed[best_index-2:best_index+3] - np.polyval([a,b,c], x[best_index-2:best_index+3])
    residuals = residuals - residuals.mean()
    rng = np.random.default_rng(42 + int(np.log10(budget)*1000))
    samples = []
    for _ in range(500):
        boot_y = observed + rng.normal(0, budget_noise, size=len(observed))
        try:
            (ba,bb,bc), _ = fit_local_isoflop([r['effective_params'] for r in group], boot_y)
        except ValueError:
            continue
        bz = -bb/(2*ba)
        bn = 10**bz
        bd = 10 ** np.interp(bz, x, np.log10([r['tokens_trained'] for r in group]))
        samples.append((bn, bd, bd/bn))
    bootstrap_optima[budget] = np.asarray(samples)
    # Count failed fits in both tails; if either endpoint is unresolved, report
    # no finite interval rather than silently conditioning on successful fits.
    failures = 500 - len(samples)
    if len(samples) >= 100 and failures < 12:
        ordered = np.sort(np.asarray(samples), axis=0)
        ci = np.vstack([ordered[12 - failures], ordered[min(len(samples)-1, 487)]])
    else:
        ci = np.full((2,3), np.nan)
    # Shift a five-point window by one depth in either direction as a sensitivity check.
    shifted = []
    shifts_evaluated = 0
    invalid_shift = False
    for start in (best_index-3, best_index-1):
        if 0 <= start and start+5 <= len(x):
            shifts_evaluated += 1
            q = np.polyfit(x[start:start+5], observed[start:start+5], 2)
            if q[0] > 0 and x[start] < -q[1]/(2*q[0]) < x[start+4]:
                shifted.append(10**(-q[1]/(2*q[0])))
            else:
                invalid_shift = True
    sensitivity = max((abs(v/n - 1) for v in shifted), default=0.0)
    budget_status[str(budget)] = {'status': 'valid', 'bootstrap_successes': len(samples),
                                  'bootstrap_failures': failures, 'bootstrap_noise_sigma': float(budget_noise),
                                  'bootstrap_noise_pool': 'regime' if regime_noise else 'all budgets',
                                  'window_shift_fraction': sensitivity, 'invalid_shift': invalid_shift,
                                  'shifts_evaluated': shifts_evaluated,
                                  'sensitive': True if invalid_shift or sensitivity > 0.1 else (False if shifts_evaluated == 2 else None)}
    optimal.append(dict(flops=budget, effective_params=n, tokens=d, ratio=d/n,
                        val_bpb=float(np.polyval([a,b,c],z)), best_measured_bpb=float(min(y)), passes=passes, repeat_factor=repeat_factor,
                        fit_min_params=10**fit_x[0], fit_max_params=10**fit_x[-1],
                        quadratic_a=a, quadratic_b=b, quadratic_c=c,
                        params_ci_low=ci[0,0], params_ci_high=ci[1,0],
                        tokens_ci_low=ci[0,1], tokens_ci_high=ci[1,1],
                        ratio_ci_low=ci[0,2], ratio_ci_high=ci[1,2],
                        bootstrap_successes=len(samples), window_shift_fraction=sensitivity))
(args.output_dir/'budget_status.json').write_text(json.dumps(budget_status, indent=2) + '\n')
if not optimal:
    raise SystemExit('No valid budget fits; see budget_status.json and extend the depth grid')
with (args.output_dir/'optimal.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(optimal[0])); writer.writeheader(); writer.writerows(optimal)
largest = max(optimal, key=lambda r: r['flops'])
selected = min((r for r in rows if r['flops_budget'] == largest['flops']),
               key=lambda r: (r['val_bpb'], r['effective_params']))
selection = {'budget': largest['flops'], 'depth': selected['depth'], 'run_name': selected['run_name'],
             'effective_params': selected['effective_params'], 'params_total': selected['params_total'],
             'fitted_optimum_params': largest['effective_params'],
             'window_shift_fraction': largest['window_shift_fraction'],
             'window_sensitive': budget_status[str(largest['flops'])]['sensitive'],
             'window_shift_invalid': budget_status[str(largest['flops'])]['invalid_shift'],
             'final_validation_bpb': selected['val_bpb'],
             'selection_rule': 'lowest final matched-validation BPB at the largest valid budget'}
(args.output_dir/'selected_model.json').write_text(json.dumps(selection, indent=2) + '\n')
fits = {}
headline = [r for r in optimal if r['flops'] <= args.headline_max_flops]
for filename, group in [('optimal_headline.csv', headline), ('optimal_beyond.csv', [r for r in optimal if r['flops'] > args.headline_max_flops])]:
    if group:
        with (args.output_dir/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(group[0])); writer.writeheader(); writer.writerows(group)
if len(headline) >= 3:
    hx = np.log10([r['flops'] for r in headline])
    ncoef = np.polyfit(hx, np.log10([r['effective_params'] for r in headline]), 1)
    dcoef = np.polyfit(hx, np.log10([r['tokens'] for r in headline]), 1)
    headline_fit = {'n': len(headline), 'params_exponent': float(ncoef[0]), 'params_coefficient': float(10**ncoef[1]), 'tokens_exponent': float(dcoef[0]), 'tokens_coefficient': float(10**dcoef[1])}
    success_fraction = float(np.prod([len(bootstrap_optima[r['flops']])/500 for r in headline]))
    headline_fit['joint_valid_fit_fraction'] = success_fraction
    if all(len(bootstrap_optima[r['flops']]) >= 100 for r in headline):
        rng = np.random.default_rng(104)
        slopes = []
        for _ in range(500):
            draw = np.array([samples[rng.integers(len(samples))] for r in headline for samples in [bootstrap_optima[r['flops']]]])
            slopes.append([np.polyfit(hx, np.log10(draw[:,0]), 1)[0], np.polyfit(hx, np.log10(draw[:,1]), 1)[0]])
        intervals = np.percentile(slopes, [2.5, 97.5], axis=0)
        headline_fit['conditional_params_exponent_ci'] = intervals[:,0].tolist()
        headline_fit['conditional_tokens_exponent_ci'] = intervals[:,1].tolist()
        headline_fit['unconditional_ci_status'] = 'resolved' if success_fraction == 1.0 else 'unresolved due to failed fits'
    fits['headline'] = headline_fit
if len(optimal) >= 3:
    all_x = np.log10([r['flops'] for r in optimal])
    fits['all_budgets'] = {
        'n': len(optimal),
        'params_exponent': float(np.polyfit(all_x, np.log10([r['effective_params'] for r in optimal]), 1)[0]),
        'tokens_exponent': float(np.polyfit(all_x, np.log10([r['tokens'] for r in optimal]), 1)[0]),
        'note': 'Combined descriptive fit; may mix fresh and repeated data regimes.',
    }
for label, group in [('fresh', [r for r in optimal if r['passes'] <= 1.0]), ('repeated', [r for r in optimal if r['passes'] > 1.0])]:
    if len(group) < 3:
        fits[label] = {'status':'insufficient budgets', 'n':len(group)}
    else:
        x=np.log10([r['flops'] for r in group])
        an, cn=np.polyfit(x,np.log10([r['effective_params'] for r in group]),1)
        ad, cd=np.polyfit(x,np.log10([r['tokens'] for r in group]),1)
        fits[label]={'n':len(group), 'params_exponent':an, 'params_coefficient':10**cn,
                     'tokens_exponent':ad, 'tokens_coefficient':10**cd,
                     'params_slope_se':float(np.sqrt(np.sum((np.log10([r['effective_params'] for r in group])-(an*x+cn))**2)/(len(group)-2)/np.sum((x-x.mean())**2))),
                     'tokens_slope_se':float(np.sqrt(np.sum((np.log10([r['tokens'] for r in group])-(ad*x+cd))**2)/(len(group)-2)/np.sum((x-x.mean())**2)))}
        if all(len(bootstrap_optima[r['flops']]) == 500 for r in group):
            rng=np.random.default_rng(104)
            slopes=[]
            for _ in range(500):
                draw=np.array([samples[rng.integers(len(samples))] for r in group for samples in [bootstrap_optima[r['flops']]]])
                slopes.append([np.polyfit(x,np.log10(draw[:,0]),1)[0], np.polyfit(x,np.log10(draw[:,1]),1)[0]])
            intervals=np.percentile(slopes,[2.5,97.5],axis=0)
            fits[label]['params_exponent_ci'] = intervals[:,0].tolist()
            fits[label]['tokens_exponent_ci'] = intervals[:,1].tolist()
with (args.output_dir/'tokens_per_param.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['flops','ratio','passes','repeat_factor']); w.writeheader()
    w.writerows({k:r[k] for k in w.fieldnames} for r in optimal)
# Use measured text exposure for the paper's unique-data and repetition terms.
if len(rows) < 8 or len({r['flops_budget'] for r in rows if r['repeat_factor'] > 1.001}) < 3:
    fits['data_constrained']={'status':'insufficient points or repeated-data observations for seven fitted constants'}
else:
    n=np.array([r['effective_params']/1e6 for r in rows]); d=np.array([r['text_tokens_trained']/1e6 for r in rows]);
    ud=np.array([r['unique_tokens']/1e6 for r in rows]); loss=np.array([r['val_bpb'] for r in rows])
    if np.any(ud <= 0) or np.any(ud > d): raise ValueError('Invalid measured data exposure')
    def predict(p):
        A,B,alpha,beta,E,Rd,Rn = p
        rd=d/ud-1
        un=np.minimum(n, (alpha*A/(beta*B))**(1/alpha) * ud**(beta/alpha))
        rn=n/un-1
        ne=un*(1+Rn*(-np.expm1(-rn/Rn)))
        de=ud*(1+Rd*(-np.expm1(-rd/Rd)))
        return A/ne**alpha+B/de**beta+E
    lower=[1e-6,1e-6,.05,.05,0,.01,.01]; upper=[100,100,3,3,10,1e4,1e4]
    best=None
    for seed in range(20):
        rng=np.random.default_rng(seed)
        p=[rng.uniform(.1,4),rng.uniform(.1,4),rng.uniform(.2,1),rng.uniform(.2,1),max(.01,float(loss.min())*.5),rng.uniform(.5,20),rng.uniform(.5,20)]
        res=least_squares(lambda x:predict(x)-loss,p,bounds=(lower,upper),max_nfev=5000)
        if best is None or res.cost < best.cost: best=res
    names=['A','B','alpha','beta','E','R_D_star','R_N_star']
    A,B,alpha,beta,E,Rd,Rn = best.x
    un = np.minimum(n, (alpha*A/(beta*B))**(1/alpha) * ud**(beta/alpha))
    condition = float(np.linalg.cond(best.jac))
    identified = (best.success and not np.any(best.active_mask) and np.isfinite(condition)
                  and condition < 1e8 and np.any(n > un * (1 + 1e-8)) and len(loss) > len(best.x))
    likelihood_ratios = {}
    if identified:
        variance = 2 * best.cost / (len(loss) - len(best.x))
        for index in (5, 6):
            free = [i for i in range(len(best.x)) if i != index]
            fixed = best.x.copy()
            fixed[index] = upper[index]
            def profile(values):
                trial = fixed.copy()
                trial[free] = values
                return predict(trial) - loss
            bound_fit = least_squares(profile, best.x[free], bounds=(np.array(lower)[free], np.array(upper)[free]), max_nfev=5000)
            delta = max(0.0, 2 * (bound_fit.cost - best.cost))
            ratio = delta / max(variance, np.finfo(float).tiny)
            likelihood_ratios[names[index]] = float(ratio)
            if not bound_fit.success or ratio <= 3.84:
                identified = False
    standard_errors = None
    if identified:
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * (2 * best.cost / (len(loss) - len(best.x)))
        standard_errors = dict(zip(names, np.sqrt(np.maximum(0, np.diag(covariance)))))
    fits['data_constrained']={'status':'fit' if identified else 'unidentified', 'parameters':dict(zip(names,best.x)),
                              'standard_errors':standard_errors, 'jacobian_condition': condition if np.isfinite(condition) else None,
                              'active_bounds':best.active_mask.tolist(),
                              'bound_likelihood_ratios': likelihood_ratios,
                              'rmse_bpb':float(np.sqrt(np.mean(best.fun**2))), 'points':len(rows),
                              'units':'N and D in millions; loss in bits per byte'}
(args.output_dir/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')

# A single two-panel figure for the model card.
plot_dir = args.output_dir / 'plots'
plot_dir.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, nx) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
for budget in sorted({r['flops_budget'] for r in rows if r['flops_budget'] <= args.headline_max_flops}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r: r['effective_params'])
    line = ax.plot([r['effective_params'] for r in group], [r['val_bpb'] for r in group], 'o', markersize=4, label=f'{budget:.3g} FLOPs')[0]
    opt = next((r for r in optimal if r['flops'] == budget), None)
    if opt:
        xx = np.linspace(np.log10(opt['fit_min_params']), np.log10(opt['fit_max_params']), 120)
        ax.plot(10**xx, np.polyval([opt['quadratic_a'], opt['quadratic_b'], opt['quadratic_c']], xx), '--', color=line.get_color())
        ax.plot(opt['effective_params'], opt['val_bpb'], '*', color=line.get_color(), markersize=10)
if selected['flops_budget'] > args.headline_max_flops:
    group = sorted((r for r in rows if r['flops_budget'] == selected['flops_budget']), key=lambda r: r['effective_params'])
    best_idx = next(i for i, r in enumerate(group) if r['run_name'] == selected['run_name'])
    local = group[max(0, best_idx-4):best_idx+5]
    ax.plot([r['effective_params'] for r in local], [r['val_bpb'] for r in local], 'o', color='#202020', markersize=4, label=f"{selected['flops_budget']:.3g} FLOPs (local)")
    xx = np.linspace(np.log10(largest['fit_min_params']), np.log10(largest['fit_max_params']), 120)
    ax.plot(10**xx, np.polyval([largest['quadratic_a'], largest['quadratic_b'], largest['quadratic_c']], xx), '--', color='#202020')
    ax.plot(selected['effective_params'], selected['val_bpb'], '*', color='#d62728', markersize=15, zorder=5)
    ax.annotate('riemann-nano', (selected['effective_params'], selected['val_bpb']), xytext=(-14, 17), textcoords='offset points', ha='right', color='#b21f2d', fontsize=9)
ax.set(xscale='log', xlabel='Effective parameters', ylabel='Final validation bits per byte', title='IsoFLOP curves')
ax.legend(fontsize=7, loc='upper right')
if headline:
    hx = np.array([r['flops'] for r in headline]); hy = np.array([r['effective_params'] for r in headline])
    nx.loglog(hx, hy, 'o', color='#2878b5', label='Fitted optima')
    fit = fits.get('headline', {})
    if 'params_exponent' in fit:
        xx = np.geomspace(hx.min(), max(hx.max(), selected['flops_budget']), 160)
        nx.loglog(xx, fit['params_coefficient']*xx**fit['params_exponent'], '--', color='#2878b5', label=f"N ∝ C^{fit['params_exponent']:.2f}")
    if selected['flops_budget'] > args.headline_max_flops:
        nx.loglog(selected['flops_budget'], selected['effective_params'], '*', color='#d62728', markersize=15)
        nx.annotate(f"riemann-nano\n{selected['effective_params']/1e6:.0f}M", (selected['flops_budget'], selected['effective_params']), xytext=(-12, 11), textcoords='offset points', ha='right', color='#b21f2d', fontsize=9)
        nx.text(.58, .12, 'Repeated data favors a larger model\nthan the headline extrapolation', transform=nx.transAxes, fontsize=8)
nx.set(xlabel='Training FLOPs', ylabel='Effective parameters', title='Optimal model size')
nx.legend(fontsize=8, loc='upper left')
fig.savefig(plot_dir / 'scaling.png')
plt.close(fig)

headline_range = f"{min(r['flops'] for r in headline):.3g}–{max(r['flops'] for r in headline):.3g}" if headline else 'no valid headline budgets'
card = ['## Scaling-law results', '', f'The headline sweep covers {headline_range} training FLOPs. Effective parameters are transformer matrices plus the language-model head. IsoFLOP curves use final-checkpoint matched-validation bits per byte; dashed quadratics fit the observed minimum and two neighbors on each side. Headline stars mark fitted minima; the red star marks the selected run.', '', '![Scaling laws](assets/scaling.png)', '']
hf = fits.get('headline', {})
if 'params_exponent' in hf:
    ni, di = hf.get('conditional_params_exponent_ci'), hf.get('conditional_tokens_exponent_ci')
    card += [f"Across these {len(headline)} budgets, N ∝ C^{hf['params_exponent']:.3f} and D ∝ C^{hf['tokens_exponent']:.3f}.", '']
    if ni and di:
        card += [f"Noise-only bootstrap 95% intervals conditional on valid fits are [{ni[0]:.3f}, {ni[1]:.3f}] for N and [{di[0]:.3f}, {di[1]:.3f}] for D. Approximately {100*hf['joint_valid_fit_fraction']:.1f}% of joint draws yield valid fits under the independent-budget assumption; unconditional intervals are {hf['unconditional_ci_status']}. These conditional intervals exclude fitting-window and scaling-model uncertainty.", '']
card += ['| FLOPs | Effective parameters (noise-only 95% CI) | Training tokens | Tokens/parameter (noise-only 95% CI) | Fitted BPB | Best measured BPB | Passes |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in headline:
    n_ci = f"{r['params_ci_low']:,.0f}–{r['params_ci_high']:,.0f}" if np.isfinite(r['params_ci_low']) else 'unresolved'
    ratio_ci = f"{r['ratio_ci_low']:.2f}–{r['ratio_ci_high']:.2f}" if np.isfinite(r['ratio_ci_low']) else 'unresolved'
    card.append(f"| {r['flops']:.3g} | {r['effective_params']:,.0f} ({n_ci}) | {r['tokens']:,.0f} | {r['ratio']:.2f} ({ratio_ci}) | {r['val_bpb']:.4f} | {r['best_measured_bpb']:.4f} | {r['passes']:.2f} |")
card += ['', '## Beyond the data', '']
beyond = [r for r in optimal if r['flops'] > args.headline_max_flops]
if beyond:
    text = f"At {min(r['flops'] for r in beyond):.3g}–{max(r['flops'] for r in beyond):.3g} FLOPs, fitted optimal training exposure ranges from {min(r['passes'] for r in beyond):.2f} to {max(r['passes'] for r in beyond):.2f} passes over the corpus."
    if {2.15e18, 3e18}.issubset({r['flops'] for r in beyond}):
        text += ' Late-run overfitting contributes to the rise and second dip in the 2.15e18 and 3e18 final-checkpoint curves; the local fitted optima are sensitive to the selected window.'
    card += [text, '']
repeat = fits['repeated']
if 'params_exponent' in repeat:
    repeated_budgets = [r['flops'] for r in optimal if r['passes'] > 1]
    card += [f"Across the repeated-data budgets ({min(repeated_budgets):.3g}–{max(repeated_budgets):.3g} FLOPs), N ∝ C^{repeat['params_exponent']:.3f} and D ∝ C^{repeat['tokens_exponent']:.3f}. This is a descriptive fit across the repeated regime.", '']
if args.diagnostics:
    diagnostics = json.loads(args.diagnostics.read_text())
    if 'beyond_overfitting' in diagnostics:
        example = diagnostics['beyond_overfitting']
        card += [f"Late-run overfitting is visible at {example['budget']:.3g} FLOPs and depth {example['depth']}: matched BPB reached {example['intermediate_min_bpb']:.6f} during training and ended at {example['final_bpb']:.6f}. All fits and selection use final checkpoints.", '']
total_text = f" and {selection['params_total']:,} total parameters" if selection['params_total'] is not None else ''
card += [f"riemann-nano is the best-measured model at the largest compute budget ({selection['budget']:.3g} FLOPs), depth {selection['depth']}. It has {selection['effective_params']:,.0f} effective parameters{total_text}, and final matched-validation BPB {selection['final_validation_bpb']:.6f}. The fitted optimum is {selection['fitted_optimum_params']:,.0f} effective parameters; shifting the fitting window changes it by up to {100*selection['window_shift_fraction']:.1f}%.", '', '## Limitations', '']
if manifest.get('train') and manifest.get('validation'):
    train, val = manifest['train'], manifest['validation']
    card += [f"The training split contains {train['files']:,} documents and {train['tokens']:,} text tokens. Matched validation contains {val['files']:,} documents. The train–validation area distribution has total-variation distance {manifest['train_validation_area_tv']:.3f}. In leaf-sampled areas, matched validation never samples non-leaf files or leaves larger than about 6% of their area; these account for approximately 61% of matched-area bytes. The effect on validation BPB is unmeasured. The three designated file-level areas are exceptions.", '']
if args.diagnostics:
    seed = diagnostics['seed_comparison']
    card += [f"{len(seed['depths'])} paired runs at {seed['budget']:.3g} FLOPs estimate {seed['noise_sigma_bpb']:.4f} BPB single-run initialization noise, with data order held fixed. Fitting-window sensitivity and failed bootstrap fits add uncertainty beyond the conditional intervals above.", '']
    if 'repeated_seed_comparison' in diagnostics:
        repeat_seed=diagnostics['repeated_seed_comparison']
        card += [f"{len(repeat_seed['depths'])} paired runs at {repeat_seed['budget']:.3g} FLOPs estimate {repeat_seed['noise_sigma_bpb']:.6f} BPB noise, also with fixed data order.", '']
if regime_noise:
    card += [f"Bootstrap noise uses the larger of seed noise and pooled local-fit residuals: {regime_noise['fresh']:.6f} BPB for the fresh regime and {regime_noise['repeated']:.6f} BPB for the repeated regime.", '']
elif args.repeated_noise_sigma is not None:
    card += [f"Fewer than three budgets in a regime prevented separate noise pooling; the bootstrap used pooled noise {noise_sigma:.4f} BPB and did not use the repeated-data floor.", '']
    print('Warning: fewer than three valid budgets in a regime; repeated-data noise floor was not used')
if args.diagnostics and 'weight_decay_comparison' in diagnostics:
    decay = diagnostics['weight_decay_comparison']
    card += [f"Reducing reference weight decay from {decay['baseline_reference_decay']:.2f} to {decay['comparison_reference_decay']:.2f} worsened the checked losses by {min(decay['bpb_differences']):.3f}–{max(decay['bpb_differences']):.3f} BPB.", '']
status = fits['data_constrained']['status']
sens = [f'{float(b):.3g}' for b, s in budget_status.items() if s.get('sensitive')]
if sens:
    card += ['Fitting-window sensitivity was detected at FLOP budgets ' + ', '.join(sens) + '.', '']
invalid = [f'{float(b):.3g}: {s["reason"]}' for b, s in budget_status.items() if s['status'] == 'invalid']
if invalid:
    card += ['Invalid budgets excluded from fits: ' + '; '.join(invalid) + '.', '']
card += [('The data-constrained seven-constant fit is unidentified; its constants should not be treated as established.' if status == 'unidentified' else f'The data-constrained seven-constant fit status is {status}.'), '', 'All scaling rows were trained on H200 GPUs. Node sizes by compute budget:', '']
for budget in sorted({row['flops_budget'] for row in rows}):
    counts = {row['nproc_per_node'] for row in rows if row['flops_budget'] == budget}
    sizes = ', '.join('unrecorded' if n is None else ('h200' if n == 1 else f'h200x{n}') for n in sorted(counts, key=lambda n: n or 0))
    card.append(f"- {budget:.3g} FLOPs: {sizes}")
card += ['', 'Token-equivalent passes divide text-token exposures by the training split text-token count.', '']
(args.output_dir/'model_card_results.md').write_text('\n'.join(card))

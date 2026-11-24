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
parser.add_argument('--noise-sigma', type=float, default=0, help='independently measured BPB noise floor (pooled with residual estimate conservatively)')
parser.add_argument('--diagnostics', type=Path, help='JSON with controlled seed and optimizer comparisons for the model card')
args = parser.parse_args()
if not np.isfinite(args.noise_sigma) or args.noise_sigma < 0:
    parser.error('--noise-sigma must be finite and nonnegative')
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
            if point in seen_points:
                raise ValueError(f'Duplicate budget/depth {point}: {seen_points[point]} and {name}')
            seen_points[point] = name
            row = {k: float(r[k]) for k in ('flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb')}
            row['run_name'] = name
            row['depth'] = int(r['depth'])
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
for budget in sorted({r['flops_budget'] for r in rows}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r: r['effective_params'])
    try:
        coeffs, _ = fit_local_isoflop([r['effective_params'] for r in group], [r['val_bpb'] for r in group])
    except ValueError:
        continue
    best = int(np.argmin([r['val_bpb'] for r in group]))
    window = group[best-2:best+3]
    residual = np.array([r['val_bpb'] for r in window]) - np.polyval(coeffs, np.log10([r['effective_params'] for r in window]))
    pooled_ss += float(residual @ residual)
    pooled_df += 2
noise_sigma = max(args.noise_sigma, np.sqrt(pooled_ss / pooled_df) if pooled_df else 0.0)
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
    observed = np.asarray(y, dtype=float)
    best_index = int(np.argmin(observed))
    residuals = observed[best_index-2:best_index+3] - np.polyval([a,b,c], x[best_index-2:best_index+3])
    residuals = residuals - residuals.mean()
    rng = np.random.default_rng(42 + int(np.log10(budget)*1000))
    samples = []
    for _ in range(500):
        boot_y = observed + rng.normal(0, noise_sigma, size=len(observed))
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
    invalid_shift = False
    for start in (best_index-3, best_index-1):
        if 0 <= start and start+5 <= len(x):
            q = np.polyfit(x[start:start+5], observed[start:start+5], 2)
            if q[0] > 0 and x[start] < -q[1]/(2*q[0]) < x[start+4]:
                shifted.append(10**(-q[1]/(2*q[0])))
            else:
                invalid_shift = True
    sensitivity = max((abs(v/n - 1) for v in shifted), default=0.0)
    budget_status[str(budget)] = {'status': 'valid', 'bootstrap_successes': len(samples),
                                  'bootstrap_failures': failures, 'bootstrap_noise_sigma': float(noise_sigma),
                                  'window_shift_fraction': sensitivity, 'invalid_shift': invalid_shift,
                                  'sensitive': bool(invalid_shift or sensitivity > 0.1)}
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
               key=lambda r: abs(np.log(r['effective_params'] / largest['effective_params'])))
selection = {'budget': largest['flops'], 'depth': selected['depth'], 'run_name': selected['run_name'],
             'effective_params': selected['effective_params'], 'fitted_optimum_params': largest['effective_params'],
             'final_validation_bpb': selected['val_bpb'],
             'selection_rule': 'nearest tested effective parameter count in log space to the fitted optimum at the largest valid budget'}
(args.output_dir/'selected_model.json').write_text(json.dumps(selection, indent=2) + '\n')
fits = {}
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

# Static files are suitable for copying into the model repository's assets/.
plot_dir=args.output_dir/'plots'; plot_dir.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':150, 'savefig.bbox':'tight', 'axes.spines.top':False, 'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7,5))
for budget in sorted({r['flops_budget'] for r in rows}):
    group=sorted((r for r in rows if r['flops_budget']==budget),key=lambda r:r['effective_params'])
    x=np.array([r['effective_params'] for r in group]); y=np.array([r['val_bpb'] for r in group])
    line=ax.plot(x,y,'o',label=f'{budget:.3g} FLOPs')[0]
    opt = next((r for r in optimal if r['flops'] == budget), None)
    if opt is None: continue
    xx=np.linspace(np.log10(opt['fit_min_params']),np.log10(opt['fit_max_params']),120)
    ax.plot(10**xx,np.polyval([opt['quadratic_a'],opt['quadratic_b'],opt['quadratic_c']],xx),color=line.get_color(),alpha=.65)
ax.set(xscale='log',xlabel='Effective parameters',ylabel='Validation bits per byte',title='IsoFLOP curves'); ax.legend(fontsize=8); fig.savefig(plot_dir/'isoflop.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r['flops'] for r in optimal],[r['val_bpb'] for r in optimal],'o-'); ax.set(xscale='log',xlabel='Training FLOPs',ylabel='Optimal fitted bits per byte',title='Fitted optimal validation loss vs compute'); fig.savefig(plot_dir/'best_loss.png'); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,ylabel,coeff,exponent in zip(axes,['effective_params','tokens'],['Optimal effective parameters','Optimal training tokens'],['params_coefficient','tokens_coefficient'],['params_exponent','tokens_exponent']):
    for label,color in [('fresh','#2878b5'),('repeated','#dc7633')]:
        group=[r for r in optimal if (r['passes']<=1.0)==(label=='fresh')]
        if not group: continue
        x=np.array([r['flops'] for r in group]); y=np.array([r[key] for r in group]); fit=fits[label]
        ax.loglog(x,y,'o',color=color,label=label)
        if 'n' in fit and coeff in fit:
            xx=np.geomspace(x.min(),x.max(),100); ax.loglog(xx,fit[coeff]*xx**fit[exponent],color=color,label=f"{label}: {fit[exponent]:.3f}")
    ax.set(xlabel='Training FLOPs',ylabel=ylabel); ax.legend(fontsize=8)
fig.suptitle('Compute-optimal allocation'); fig.tight_layout(); fig.savefig(plot_dir/'optimal_nd.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r['flops'] for r in optimal],[r['ratio'] for r in optimal],'o-'); ax.set(xscale='log',xlabel='Training FLOPs',ylabel='Tokens per effective parameter',title='Optimal tokens:parameters'); fig.savefig(plot_dir/'tokens_per_param.png'); plt.close(fig)
area_plot=False
if args.area_dir:
    area_files=sorted(args.area_dir.glob('*.json'))
    valid_names = {r['run_name'] for r in rows}
    payloads=[p for p in (json.loads(path.read_text()) for path in area_files) if p.get('checkpoint') in valid_names]
    def checkpoint_order(payload):
        match = re.search(r'_([0-9.]+e[+-]?[0-9]+)_d([0-9]+)$', payload['checkpoint'], re.I)
        return (float(match.group(1)), int(match.group(2))) if match else (float('inf'), 0)
    payloads.sort(key=checkpoint_order)
    expected_areas = set(manifest.get('file_level_areas', {}))
    if any(set(p.get('file_level_areas', {})) != expected_areas or p.get('tokenizer_sha256') != expected_sha
           or p.get('area_evaluation_set') != manifest['area_evaluation_set'] for p in payloads):
        raise ValueError('Area results do not match the canonical tokenizer and split')
    if payloads:
        areas=sorted(manifest['area_split_bytes'])
        matrix=np.array([[p['areas'].get(a,{}).get('bits_per_byte',np.nan) for p in payloads] for a in areas],dtype=float)
        fig,ax=plt.subplots(figsize=(max(7,len(payloads)*.45),max(5,min(60,len(areas)*.18))))
        im=ax.imshow(matrix,aspect='auto',interpolation='nearest',vmax=np.nanpercentile(matrix,99)); ax.set_yticks(range(len(areas)),[a + (' [' + manifest['area_evaluation_set'][a] + ']') + (' *' if a in expected_areas else '') for a in areas],fontsize=4); ax.set_xticks(range(len(payloads)),[p['checkpoint'] for p in payloads],rotation=90,fontsize=6); ax.set(title='Held-out difficulty (unseen = transfer; * file-level split)',xlabel='Checkpoint'); fig.colorbar(im,ax=ax,label='Bits per byte (99th percentile cap)'); fig.savefig(plot_dir/'directory_difficulty.png'); plt.close(fig); area_plot=True
card=['## Scaling-law results','','We trained on the Lean corpus with held-out source directories. The figures report validation bits per byte, training compute and effective parameters (transformer matrices plus the language-model head).','',
      '![IsoFLOP curves](assets/isoflop.png)','','![Best loss against compute](assets/best_loss.png)','','![Optimal model size and training tokens](assets/optimal_nd.png)','','![Optimal tokens per parameter](assets/tokens_per_param.png)','']
card += [f"The selected checkpoint is depth {selection['depth']} at {selection['budget']:.3g} training FLOPs, the tested model closest in log parameter count to the fitted optimum at the largest valid budget. Its final-checkpoint matched-validation loss is {selection['final_validation_bpb']:.4f} bits per byte.", '']
if manifest.get('train') and manifest.get('validation'):
    train, val = manifest['train'], manifest['validation']
    card += [f"The training split contains {train['files']:,} documents and {train['tokens']:,} text tokens. Matched validation contains {val['files']:,} documents. The train–validation area distribution has total-variation distance {manifest['train_validation_area_tv']:.3f}.", '']
    card += ['Matched validation samples eligible whole leaves, so it underrepresents large leaves and files outside leaf directories within an area. Matching the area distribution does not remove this within-area sampling limitation.', '']
if args.diagnostics:
    diagnostics = json.loads(args.diagnostics.read_text())
    seed = diagnostics['seed_comparison']
    decay = diagnostics['weight_decay_comparison']
    card += [f"A second initialization seed at {seed['budget']:.3g} FLOPs across {len(seed['depths'])} depths gave a provisional {seed['noise_sigma_bpb']:.4f} BPB single-run noise estimate; data order was unchanged. Reducing the reference weight decay from {decay['baseline_reference_decay']:.2f} to {decay['comparison_reference_decay']:.2f} worsened those depths by {min(decay['bpb_differences']):.3f}–{max(decay['bpb_differences']):.3f} BPB. The fitted scaling behavior is conditional on the optimizer schedule.", '']
    if 'batch_comparison' in diagnostics:
        batch = diagnostics['batch_comparison']
        card += [f"At {batch['budget']:.3g} FLOPs and depth {batch['depth']}, an efficient batch changed BPB from {batch['baseline_bpb']:.6f} to {batch['comparison_bpb']:.6f}. This comparison also slightly changed total batch and optimizer scaling, so it does not isolate numerical rounding.", '']
if area_plot: card += ['![Difficulty by source area](assets/directory_difficulty.png)','']
else: card += ['Directory difficulty plot pending per-checkpoint evaluation.','']
for regime in ('fresh','repeated'):
    fit=fits[regime]
    if 'params_exponent' in fit:
        intervals = ''
        if 'params_exponent_ci' in fit:
            ni, di = fit['params_exponent_ci'], fit['tokens_exponent_ci']
            intervals = f" Noise-only bootstrap 95% intervals are [{ni[0]:.3f}, {ni[1]:.3f}] and [{di[0]:.3f}, {di[1]:.3f}], respectively; they exclude uncertainty from the fitting-window choice."
        card += [f"For {regime} data, the fitted exponents are N ∝ C^{fit['params_exponent']:.3f} and D ∝ C^{fit['tokens_exponent']:.3f}." + intervals, '']
card += ['', 'Regime labels use interpolated passes at the optimum; individual models in a fit window can fall on the other side of one pass.', '']
card += ['| FLOPs | Effective parameters (noise-only 95% CI) | Training tokens | Tokens/parameter (noise-only 95% CI) | Fitted BPB | Best measured BPB | Passes |','| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in optimal:
    n_ci = f"{r['params_ci_low']:,.0f}–{r['params_ci_high']:,.0f}" if np.isfinite(r['params_ci_low']) else 'unresolved'
    ratio_ci = f"{r['ratio_ci_low']:.2f}–{r['ratio_ci_high']:.2f}" if np.isfinite(r['ratio_ci_low']) else 'unresolved'
    card.append(f"| {r['flops']:.3g} | {r['effective_params']:,.0f} ({n_ci}) | {r['tokens']:,.0f} | {r['ratio']:.2f} ({ratio_ci}) | {r['val_bpb']:.4f} | {r['best_measured_bpb']:.4f} | {r['passes']:.2f} |")
sensitive = [f'{float(b):.3g}' for b, status in budget_status.items() if status.get('sensitive')]
if sensitive:
    card += ['', 'The fitted optimum is sensitive to a one-depth fitting-window shift (more than 10% change, or an invalid shifted fit) at FLOP budgets ' + ', '.join(sensitive) + '; regime exponents that include these budgets inherit this sensitivity.']
if any(not np.isfinite(r['params_ci_low']) for r in optimal):
    card += ['', 'Intervals are unresolved where too many bootstrap draws fail the fit; failed draws are not discarded from the confidence calculation.']
invalid = [f'{float(b):.3g}: {s["reason"]}' for b, s in budget_status.items() if s['status'] == 'invalid']
if invalid:
    card += ['', 'Invalid budgets excluded from fits: ' + '; '.join(invalid) + '.']
card += ['', f"Data-constrained fit status: {fits['data_constrained']['status']}. Repeated-data exponent status: {fits['repeated'].get('status', 'fit')}."]
card += ['', f"Token-equivalent passes divide text-token exposures by the training split text-token count. Reuse factors and measured unique token positions are recorded in the results data. {len(manifest.get('file_level_areas', {}))} areas use file-level validation splits. Matched validation measures held-out leaves from training areas; difficulty-only leaves and unseen areas are excluded from scaling fits, and unseen areas measure transfer.",'']
(args.output_dir/'model_card_results.md').write_text('\n'.join(card))

"""Analyze RH results. Run: uv run --extra cpu --with scipy --with matplotlib python -m scripts.analyze_rh_scaling --results ... --output-dir ...

The fitted loss follows Muennighoff et al., https://arxiv.org/abs/2305.16264,
with N and D expressed in millions; bpB is the fitted loss. Estimates from
unbracketed minima and underdetermined fits are explicitly rejected.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from nanochat.scaling import fit_local_isoflop

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, nargs='+', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--area-dir', type=Path, help='directory of per-checkpoint area evaluation JSON files')
parser.add_argument('--asset-prefix', default='rh_v2', help='only analyze runs using this dataset and tokenizer version')
parser.add_argument('--head-dim', type=int, default=64)
args = parser.parse_args()
rows = []
for path in args.results:
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get('asset_prefix') != args.asset_prefix or r.get('head_dim') != str(args.head_dim):
                raise ValueError(f'{path}: result does not match asset prefix {args.asset_prefix!r} and head dimension {args.head_dim}; do not mix experiments')
            if int(r['model_dim']) != int(r['depth']) * 64 or int(r['num_iterations']) < 500:
                raise ValueError(f'{path}: incompatible model width or fewer than 500 optimizer steps')
            rows.append({k: float(r[k]) for k in ('flops_budget', 'effective_params', 'tokens_trained', 'text_tokens_trained', 'unique_tokens', 'passes', 'repeat_factor', 'val_bpb')})
if not rows:
    raise SystemExit('No results')
args.output_dir.mkdir(parents=True, exist_ok=True)
optimal = []
for budget in sorted({r['flops_budget'] for r in rows}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r:r['effective_params'])
    x = np.log10([r['effective_params'] for r in group]); y = [r['val_bpb'] for r in group]
    try:
        (a,b,c), fit_x = fit_local_isoflop([r['effective_params'] for r in group], y)
    except ValueError as error:
        raise ValueError(f'{budget:g}: {error}') from error
    z = -b/(2*a); n = 10**z
    d = float(np.interp(z,x,[r['tokens_trained'] for r in group]))
    passes = float(np.interp(z,x,[r['passes'] for r in group]))
    repeat_factor = float(np.interp(z,x,[r['repeat_factor'] for r in group]))
    optimal.append(dict(flops=budget, effective_params=n, tokens=d, ratio=d/n,
                        val_bpb=float(np.polyval([a,b,c],z)), passes=passes, repeat_factor=repeat_factor,
                        fit_min_params=10**fit_x[0], fit_max_params=10**fit_x[-1],
                        quadratic_a=a, quadratic_b=b, quadratic_c=c))
with (args.output_dir/'optimal.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(optimal[0])); writer.writeheader(); writer.writerows(optimal)
fits = {}
for label, group in [('fresh', [r for r in optimal if r['repeat_factor'] <= 1.001]), ('repeated', [r for r in optimal if r['repeat_factor'] > 1.001])]:
    if len(group) < 2:
        fits[label] = {'status':'insufficient budgets', 'n':len(group)}
    else:
        x=np.log10([r['flops'] for r in group])
        an, cn=np.polyfit(x,np.log10([r['effective_params'] for r in group]),1)
        ad, cd=np.polyfit(x,np.log10([r['tokens'] for r in group]),1)
        fits[label]={'n':len(group), 'params_exponent':an, 'params_coefficient':10**cn,
                     'tokens_exponent':ad, 'tokens_coefficient':10**cd}
with (args.output_dir/'tokens_per_param.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['flops','ratio','passes','repeat_factor']); w.writeheader()
    w.writerows({k:r[k] for k in w.fieldnames} for r in optimal)
# Use measured text exposure for the paper's unique-data and repetition terms.
if len(rows) < 8 or sum(r['repeat_factor'] > 1.001 for r in rows) < 3:
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
    fits['data_constrained']={'status':'fit' if best.success else 'optimizer failed', 'parameters':dict(zip(names,best.x)),
                              'rmse_bpb':float(np.sqrt(np.mean(best.fun**2))), 'points':len(rows),
                              'units':'N and D in millions; loss in bits per byte',
                              'warning':'No uncertainty estimate; fit may be weakly identified.'}
(args.output_dir/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')

# Static files are suitable for copying into the model repository's assets/.
plot_dir=args.output_dir/'plots'; plot_dir.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':150, 'savefig.bbox':'tight', 'axes.spines.top':False, 'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7,5))
for budget in sorted({r['flops_budget'] for r in rows}):
    group=sorted((r for r in rows if r['flops_budget']==budget),key=lambda r:r['effective_params'])
    x=np.array([r['effective_params'] for r in group]); y=np.array([r['val_bpb'] for r in group])
    line=ax.plot(x,y,'o',label=f'{budget:.0e} FLOPs')[0]
    opt = next(r for r in optimal if r['flops'] == budget)
    xx=np.linspace(np.log10(opt['fit_min_params']),np.log10(opt['fit_max_params']),120)
    ax.plot(10**xx,np.polyval([opt['quadratic_a'],opt['quadratic_b'],opt['quadratic_c']],xx),color=line.get_color(),alpha=.65)
ax.set(xscale='log',xlabel='Effective parameters',ylabel='Validation bits per byte',title='IsoFLOP curves'); ax.legend(fontsize=8); fig.savefig(plot_dir/'isoflop.png'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4)); ax.plot([r['flops'] for r in optimal],[r['val_bpb'] for r in optimal],'o-'); ax.set(xscale='log',xlabel='Training FLOPs',ylabel='Optimal fitted bits per byte',title='Best validation loss vs compute'); fig.savefig(plot_dir/'best_loss.png'); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,ylabel,coeff,exponent in zip(axes,['effective_params','tokens'],['Optimal effective parameters','Optimal training tokens'],['params_coefficient','tokens_coefficient'],['params_exponent','tokens_exponent']):
    for label,color in [('fresh','#2878b5'),('repeated','#dc7633')]:
        group=[r for r in optimal if (r['repeat_factor']<=1.001)==(label=='fresh')]
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
    payloads=[json.loads(p.read_text()) for p in area_files]
    if args.asset_prefix == 'rh_v2':
        expected = {'(root)', 'Formalism', 'Development/ResolutionOfSingularities'}
        if any(set(p.get('file_level_areas', {})) != expected for p in payloads):
            raise ValueError('Area results do not match the revised split; do not mix old and revised evaluations')
    if payloads:
        areas=sorted({name for p in payloads for name in p['areas']})
        matrix=np.array([[p['areas'].get(a,{}).get('bits_per_byte',np.nan) for p in payloads] for a in areas],dtype=float)
        fig,ax=plt.subplots(figsize=(max(7,len(payloads)*.45),max(5,min(60,len(areas)*.18))))
        im=ax.imshow(matrix,aspect='auto',interpolation='nearest'); ax.set_yticks(range(len(areas)),[a + (' *' if any(a in p.get('file_level_areas', {}) for p in payloads) else '') for a in areas],fontsize=4); ax.set_xticks(range(len(payloads)),[p['checkpoint'] for p in payloads],rotation=90,fontsize=6); ax.set(title='Held-out difficulty by source area (* file-level split)',xlabel='Checkpoint'); fig.colorbar(im,ax=ax,label='Bits per byte'); fig.savefig(plot_dir/'directory_difficulty.png'); plt.close(fig); area_plot=True
card=['## Scaling-law results','','We trained on the Lean corpus with held-out source directories. The figures report validation bits per byte, effective parameters and training compute.','',
      '![IsoFLOP curves](assets/isoflop.png)','','![Best loss against compute](assets/best_loss.png)','','![Optimal model size and training tokens](assets/optimal_nd.png)','','![Optimal tokens per parameter](assets/tokens_per_param.png)','']
if area_plot: card += ['![Difficulty by source area](assets/directory_difficulty.png)','']
else: card += ['Directory difficulty plot pending per-checkpoint evaluation.','']
for regime in ('fresh','repeated'):
    fit=fits[regime]
    if 'params_exponent' in fit:
        card += [f"For {regime} data, the fitted exponents are N ∝ C^{fit['params_exponent']:.3f} and D ∝ C^{fit['tokens_exponent']:.3f}.", '']
card += ['| FLOPs | Effective parameters | Training tokens | Tokens/parameter | Validation BPB | Token-equivalent passes |','| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in optimal:
    card.append(f"| {r['flops']:.1e} | {r['effective_params']:,.0f} | {r['tokens']:,.0f} | {r['ratio']:.2f} | {r['val_bpb']:.4f} | {r['passes']:.2f} |")
card += ['', 'Token-equivalent passes divide text-token exposures by the full corpus text-token count. Reuse factors and measured unique token positions are recorded in the results data. Three areas use file-level validation splits; other areas use directory-level splits.','']
(args.output_dir/'model_card_results.md').write_text('\n'.join(card))

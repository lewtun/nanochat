"""Analyze RH results. Run: uv run --extra cpu --with scipy python -m scripts.analyze_rh_scaling --results ... --output-dir ...

The fitted loss follows Muennighoff et al., https://arxiv.org/abs/2305.16264,
with N and D expressed in millions; bpB is the fitted loss. Estimates from
unbracketed minima and underdetermined fits are explicitly rejected.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, nargs='+', required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
rows = []
for path in args.results:
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(r[k]) for k in ('flops_budget', 'effective_params', 'tokens_trained', 'passes', 'val_bpb')})
if not rows:
    raise SystemExit('No results')
args.output_dir.mkdir(parents=True, exist_ok=True)
optimal = []
for budget in sorted({r['flops_budget'] for r in rows}):
    group = sorted((r for r in rows if r['flops_budget'] == budget), key=lambda r:r['effective_params'])
    if len(group) < 3:
        raise ValueError(f'{budget:g}: need at least three distinct model sizes')
    x = np.log10([r['effective_params'] for r in group]); y = [r['val_bpb'] for r in group]
    if len(set(x)) != len(x): raise ValueError(f'{budget:g}: duplicate parameter counts')
    a,b,c = np.polyfit(x,y,2)
    if a <= 0 or not x[0] < -b/(2*a) < x[-1]:
        raise ValueError(f'{budget:g}: fitted minimum is not bracketed')
    z = -b/(2*a); n = 10**z
    d = float(np.interp(z,x,[r['tokens_trained'] for r in group]))
    passes = float(np.interp(z,x,[r['passes'] for r in group]))
    optimal.append(dict(flops=budget, effective_params=n, tokens=d, ratio=d/n,
                        val_bpb=float(np.polyval([a,b,c],z)), passes=passes))
with (args.output_dir/'optimal.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(optimal[0])); writer.writeheader(); writer.writerows(optimal)
fits = {}
for label, group in [('fresh', [r for r in optimal if r['passes'] <= 1]), ('repeated', [r for r in optimal if r['passes'] > 1])]:
    if len(group) < 2:
        fits[label] = {'status':'insufficient budgets', 'n':len(group)}
    else:
        x=np.log10([r['flops'] for r in group])
        an, cn=np.polyfit(x,np.log10([r['effective_params'] for r in group]),1)
        ad, cd=np.polyfit(x,np.log10([r['tokens'] for r in group]),1)
        fits[label]={'n':len(group), 'params_exponent':an, 'params_coefficient':10**cn,
                     'tokens_exponent':ad, 'tokens_coefficient':10**cd}
with (args.output_dir/'tokens_per_param.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['flops','ratio','passes']); w.writeheader()
    w.writerows({k:r[k] for k in w.fieldnames} for r in optimal)
# D_c is the training corpus token count; passes in the input are D / D_c.
if len(rows) < 8:
    fits['data_constrained']={'status':'insufficient points for seven fitted constants'}
else:
    n=np.array([r['effective_params']/1e6 for r in rows]); d=np.array([r['tokens_trained']/1e6 for r in rows]);
    dc=np.array([r['tokens_trained']/r['passes']/1e6 for r in rows]); loss=np.array([r['val_bpb'] for r in rows])
    if not np.allclose(dc,dc[0],rtol=1e-6): raise ValueError('Inconsistent corpus token counts')
    def predict(p):
        A,B,alpha,beta,E,Rd,Rn = p
        ud=np.minimum(d,dc); rd=d/ud-1
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

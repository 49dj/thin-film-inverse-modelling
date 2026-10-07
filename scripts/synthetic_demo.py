"""Estimate one known-physics thickness; full multi-parameter fit is optional."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,differential_evolution
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from synthetic_spectra import synthetic_spectra
from thinfilm.objective import build_model_curves_ideal
from thinfilm.optimisation import fit_silicon
from thinfilm.plotting import plot_dual_angle

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full-fit',action='store_true',help='expensive Q3 multi-parameter DE+LS workflow')
    ap.add_argument('--output',type=Path,default=Path('results'))
    args=ap.parse_args()
    sigma,data,truth=synthetic_spectra()
    if args.full_fit:
        de,ls=fit_silicon(data,sigma,5.)
        fitted=ls.x
    else:
        # New reduced demo objective: fixed nuisance parameters, no thickness penalty.
        def residual(x):
            p=truth.copy();p[0]=x[0]
            return np.concatenate([fit-y for (_,y),fit in zip(data,build_model_curves_ideal(p,data,sigma))])
        de=differential_evolution(lambda x:float(residual(x)@residual(x)),[(3.,8.)],seed=42,maxiter=40,popsize=8,polish=False)
        ls=least_squares(residual,de.x,bounds=([3.],[8.]),xtol=1e-12,gtol=1e-12,ftol=1e-12)
        fitted=truth.copy();fitted[0]=ls.x[0]
    curves=build_model_curves_ideal(fitted,data,sigma)
    rmse=float(np.sqrt(np.mean(np.concatenate([fit-y for (_,y),fit in zip(data,curves)])**2)))
    report={'synthetic':True,'reduced_demo':not args.full_fit,'known_thickness_um':float(truth[0]),
        'estimated_thickness_um':float(fitted[0]),'rmse':rmse,'de_success':bool(de.success),'ls_success':bool(ls.success)}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'fit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    plot_dual_angle(sigma,data,curves,args.output/'dual_angle.png')
    print(json.dumps(report))
    if not np.isfinite(rmse):raise RuntimeError('Nonfinite output')

if __name__=='__main__':main()

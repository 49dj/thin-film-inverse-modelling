"""Explicit wrapper around the Q3 DE SSE / soft-L1 least-squares sequence."""
import numpy as np
from scipy.optimize import least_squares, differential_evolution
from .objective import residual_vector, de_objective, bounds_and_init

def fit_silicon(datasets,sigma,d0,de_iterations=100,popsize=10,max_nfev=1800,seed=42,polish=True):
    p0,lb,ub=bounds_and_init(d0,datasets)
    coarse_sigma=sigma[::2]
    coarse_data=[(angle,np.interp(coarse_sigma,sigma,y)) for angle,y in datasets]
    de=differential_evolution(de_objective,list(zip(lb,ub)),
        args=(coarse_data,coarse_sigma,d0),strategy='best1bin',maxiter=de_iterations,
        popsize=popsize,tol=1e-6,mutation=(0.5,1.),recombination=0.7,
        polish=polish,updating='deferred',workers=1,seed=seed)
    ls=least_squares(lambda p:residual_vector(p,datasets,sigma,d0),de.x,
        bounds=(lb,ub),loss='soft_l1',f_scale=0.03,max_nfev=max_nfev,
        xtol=1e-10,ftol=1e-10,gtol=1e-10)
    return de,ls

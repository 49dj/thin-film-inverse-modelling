"""Dual-angle silicon spectra created from the retained Q3 forward equations."""
import numpy as np
from thinfilm.objective import bounds_and_init,build_model_curves_ideal

def synthetic_spectra(n=100,noise=0.,seed=2026):
    sigma=np.linspace(1200.,3200.,n)
    datasets=[(10.,np.zeros(n)),(15.,np.zeros(n))]
    params,_,_=bounds_and_init(5.,datasets)
    params[0]=5.4
    curves=build_model_curves_ideal(params,datasets,sigma)
    rng=np.random.default_rng(seed)
    return sigma,[(angle,y+rng.normal(0,noise,n)) for (angle,_),y in zip(datasets,curves)],params

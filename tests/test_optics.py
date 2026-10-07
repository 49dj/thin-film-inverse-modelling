import numpy as np
import pytest
from thinfilm.forward_model import reflectance_film_oblique,reflectance_bulk,wnum_cm_to_lambda_m
from thinfilm.silicon import reflectance_abeles
from thinfilm.objective import residual_vector,bounds_and_init,build_model_curves_ideal
from thinfilm.optimisation import fit_silicon

def test_zero_thickness_is_direct_interface():
    sigma=np.linspace(1200,3000,40)
    result=reflectance_film_oblique(np.full(40,4.+0j),np.full(40,9.+0j),0.,wnum_cm_to_lambda_m(sigma),0.)
    assert np.allclose(result,reflectance_bulk(np.full(40,9.)))
    abeles=reflectance_abeles(sigma,0.,[np.ones(40),np.full(40,2.),np.full(40,3.)],[0.])
    assert np.allclose(abeles,result)

def test_q3_exact_synthetic_residual():
    sigma=np.linspace(1200,3200,60);data=[(10.,np.zeros(60)),(15.,np.zeros(60))]
    p,lb,ub=bounds_and_init(5.,data)
    curves=build_model_curves_ideal(p,data,sigma)
    data=[(angle,y) for (angle,_),y in zip(data,curves)]
    assert np.all((lb<=p)&(p<=ub))
    assert residual_vector(p,data,sigma,5.).shape==(130,)
    assert np.allclose(residual_vector(p,data,sigma,5.),0.)

def test_full_objective_optimizer_smoke():
    sigma=np.linspace(1200,3200,20);data=[(10.,np.zeros(20)),(15.,np.zeros(20))]
    p,lb,ub=bounds_and_init(5.,data)
    data=[(angle,y) for (angle,_),y in zip(data,build_model_curves_ideal(p,data,sigma))]
    de,ls=fit_silicon(data,sigma,5.,de_iterations=1,popsize=1,max_nfev=2,polish=False)
    assert np.isfinite(de.fun) and np.isfinite(ls.cost)
    assert np.all((ls.x>=lb)&(ls.x<=ub))
    # Iteration-limited run tests execution; it does not assert convergence.

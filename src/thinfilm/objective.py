import numpy as np
from .silicon import ideal_reflectance_with_averaging

def spectral_weights_for_fit(sigma):
    w = np.ones_like(sigma, float)
    w[(sigma>=700) & (sigma<=1100)] *= 4.0
    w[(sigma>=1500)] *= 2.0      # 可选：增强尾部拟合权重
    return w

def build_model_curves_ideal(params, datasets, sigma, coarse=False):
    (d,t,Ne,mue,Ns,mus,eps_e,eps_s,S1,w1,g1,S2,w2,g2,A,Bd,Bw,sigth,d_jit)=params[:19]
    nA,nD,nT=(5 if coarse else 7),(2 if coarse else 3),(2 if coarse else 3)
    curves=[]
    for(ang,_),sc in zip(datasets,params[19:19+len(datasets)]):
        R0=ideal_reflectance_with_averaging(
            sigma,ang,d,t,Ne,mue,Ns,mus,eps_e,eps_s,[S1,w1,g1,S2,w2,g2],
            A,Bd,Bw,nA,max(sigth,1e-3),nD,max(d_jit,0.),nT)
        curves.append(sc*np.clip(R0,-0.2,1.2))
    return curves

def residual_vector(p,datasets,sigma,d0,coarse=False):
    w=spectral_weights_for_fit(sigma); curves=build_model_curves_ideal(p,datasets,sigma,coarse=coarse)
    res=np.concatenate([(Rm-Ry)*w for(_,Ry),Rm in zip(datasets,curves)])
    res=np.concatenate([res,0.1*(p[0]-d0)*np.ones(10)]); return res

def objective_sse(p,datasets,sigma,d0,coarse=False):
    r = residual_vector(p,datasets,sigma,d0,coarse=coarse); return float(np.dot(r, r))

def bounds_and_init(d0,datasets):
    p0=[d0,2.,5e18,50.,8e19,300.,11.7,11.7,0.3,1070.,40.,0.15,800.,60.,0.1,0.01,500.,0.4,0.03]
    for _ in datasets: p0+=[1.]
    lb=[d0*0.6,0.,1e17,5.,1e19,10.,10.5,10.5,0.05,1030.,20.,0.05,700.,20.,0.,0.0,100.,0.02,0.]
    for _ in datasets: lb+=[0.6]
    ub=[d0*1.6,6.,5e19,600.,5e20,800.,13.3,13.3,1.5,1135.,90.,0.6,900.,120.,0.8,0.5,2000.,2.,0.2]
    for _ in datasets: ub+=[1.8]
    return np.array(p0,float),np.array(lb,float),np.array(ub,float)

def de_objective(p, datasets_coarse, sigma_coarse, d0):
    return objective_sse(np.array(p,float), datasets_coarse, sigma_coarse, d0, coarse=True)

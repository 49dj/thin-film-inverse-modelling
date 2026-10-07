import numpy as np
e=1.602176634e-19
eps0=8.8541878128e-12
m_e=9.10938356e-31
c_cm_s=2.99792458e10
TWOPI=2*np.pi
SIGMA_REF=1000.0
HF_CFG={'thr':700.0,'cap':1.2,'shape':1.0}

def drude_eps_Si(sigma_cm1, N_cm3, mu_cm2Vs, eps_inf=11.7, mstar=0.26):
    w=TWOPI*c_cm_s*sigma_cm1; m_eff=mstar*m_e; N_m3=max(N_cm3,1e10)*1e6
    mu_m2=max(mu_cm2Vs,1e-6)*1e-4; wp2=N_m3*e*e/(eps0*m_eff); gamma=e/(m_eff*mu_m2)
    return eps_inf - wp2/(w*(w+1j*gamma))

def lorentz_eps_SiO2(sigma_cm1,eps_inf=2.1,S1=0.3,w01=1070.,g1=40.,S2=0.15,w02=800.,g2=60.,g_min=8.):
    s=sigma_cm1.astype(float); g1e=max(g1,g_min); g2e=max(g2,g_min)
    t1=S1*w01**2/((w01**2-s*s)-1j*g1e*s); t2=S2*w02**2/((w02**2-s*s)-1j*g2e*s)
    return eps_inf + t1 + t2

def si_extra_dispersion_real(sigma_cm1, A_disp=0.10):
    return A_disp*(SIGMA_REF/np.maximum(sigma_cm1,1.0))**4

def _sat_log_weight(x, cap=1.2, shape=1.0):
    z = np.log1p(np.maximum(x, 0.0))
    y = 1.0 - np.exp(- (z / max(cap, 1e-9))**max(shape, 1e-6))
    return y

def eps_Si_total(sigma, N, mu, eps_inf, A_disp, B_damp, B_width):
    base_eps = drude_eps_Si(sigma, N, mu, eps_inf)
    low_freq_disp = si_extra_dispersion_real(sigma, A_disp)
    x = (sigma - HF_CFG['thr']) / max(B_width, 1e-9)
    w_sat = _sat_log_weight(x, cap=HF_CFG['cap'], shape=HF_CFG['shape'])
    high_freq_correction = -B_damp * w_sat
    return base_eps + low_freq_disp + high_freq_correction

def n_from_eps(eps): return np.sqrt(eps + 0j)

def _angles_in_stack(n_list, theta0_deg):
    th0=np.deg2rad(theta0_deg); n0=n_list[0]; th=[th0]
    for j in range(1,len(n_list)): th.append(np.arcsin(n0*np.sin(th0)/n_list[j]))
    return th

def _q(n, th, pol):
    ct=np.cos(th); return n*ct if pol=='s' else n/np.cos(th)

def reflectance_abeles(sigma_cm1, theta0_deg, n_list, d_list_um, pol='avg'):
    L=len(n_list)-2; th=_angles_in_stack(n_list, theta0_deg)
    def calc(polx):
        q0=_q(n_list[0],th[0],polx); qS=_q(n_list[-1],th[-1],polx)
        M11=np.ones_like(sigma_cm1,complex); M12=np.zeros_like(sigma_cm1,complex)
        M21=np.zeros_like(sigma_cm1,complex); M22=np.ones_like(sigma_cm1,complex)
        for j in range(1,L+1):
            nj,thj=n_list[j],th[j]; dj_cm=d_list_um[j-1]*1e-4
            delta=TWOPI*sigma_cm1*nj*np.cos(thj)*dj_cm
            c,s=np.cos(delta),np.sin(delta); qj=_q(nj,thj,polx)
            A11,A12,A21,A22=c,1j*s/qj,1j*qj*s,c
            T11,T12,T21,T22=M11*A11+M12*A21,M11*A12+M12*A22,M21*A11+M22*A21,M21*A12+M22*A22
            M11,M12,M21,M22 = T11,T12,T21,T22
        num=(q0*M11+q0*qS*M12-M21-qS*M22); den=(q0*M11+q0*qS*M12+M21+qS*M22)
        den=np.where(np.abs(den)<1e-30,1e-30+0j,den); r=num/den; return np.real(r*np.conj(r))
    Rs=calc('s'); Rp=calc('p'); return 0.5*(Rs+Rp) if pol=='avg' else (Rs if pol=='s' else Rp)

def _gauss_nodes(mu, sig, n):
    if sig<=0 or n<=1: return np.array([mu],float), np.array([1.0],float)
    xs=np.linspace(mu-3*sig,mu+3*sig,n); w=np.exp(-0.5*((xs-mu)/sig)**2); w/=w.sum(); return xs,w

def ideal_reflectance_with_averaging(
    sigma, angle_deg, d_um, tox_nm, N_e, mu_e, N_s, mu_s,
    eps_inf_e, eps_inf_s, lorentz_params, A_disp, B_damp, B_width,
    n_samples_angle=7, sigma_theta0_deg=0.40, n_samples_d=3, d_jitter_um=0.03, n_samples_tox=3):
    S1,w01,g1,S2,w02,g2=lorentz_params
    eps_ox=lorentz_eps_SiO2(sigma,2.1,S1,w01,g1,S2,w02,g2)
    eps_e=eps_Si_total(sigma,N_e,mu_e,eps_inf_e,A_disp,B_damp,B_width)
    eps_s=eps_Si_total(sigma,N_s,mu_s,eps_inf_s,A_disp,0.0,1000.0)  # 衬底不施加高频饱和修正
    n_air=np.ones_like(sigma,complex); n_list=[n_air, n_from_eps(eps_ox), n_from_eps(eps_e), n_from_eps(eps_s)]
    angs,wa=_gauss_nodes(angle_deg,sigma_theta0_deg,n_samples_angle)
    dvals,wd=_gauss_nodes(d_um,d_jitter_um,n_samples_d); toxvals,wt=_gauss_nodes(tox_nm,0.2,n_samples_tox)
    R=np.zeros_like(sigma,float); W=0.0
    for th,wth in zip(angs,wa):
      for dd,wd_i in zip(dvals,wd):
        for tt,wt_i in zip(toxvals,wt):
          d_list=[tt*1e-3, dd]
          R+=wth*wd_i*wt_i*reflectance_abeles(sigma,th,n_list,d_list,'avg'); W+=wth*wd_i*wt_i
    return R/max(W,1e-12)

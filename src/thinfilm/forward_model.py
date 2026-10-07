import math
import numpy as np
e_charge = 1.602176634e-19
eps0 = 8.8541878128e-12
m_e = 9.10938356e-31
c_cm_per_s = 2.99792458e10

def m_eff_4H_concentration(m_MK=0.31, m_MG=0.58):
    return math.sqrt(m_MK * m_MG) * m_e

def m_eff_4H_mobility(m_MK=0.31, m_MG=0.58):
    num = (m_MK ** (-0.5) + m_MG ** (-0.5))
    den = (m_MK ** (-1.5) + m_MG ** (-1.5))
    return (num / den) * m_e

def m_eff_6H_isotropic():
    return 0.35 * m_e

def wnum_cm_to_omega(wnum_cm): return 2.0 * np.pi * c_cm_per_s * wnum_cm

def wnum_cm_to_lambda_m(wnum_cm):
    sigma_m = wnum_cm * 100.0
    return 1.0 / sigma_m

def eps_mdf(omega, eps_inf, wT, wL, GT, GL, w_p, gamma_p):
    omega = np.asarray(omega, dtype=np.complex128)
    num = (wL**2 - omega**2 - 1j * GL * omega)
    den = (wT**2 - omega**2 - 1j * GT * omega)
    phonon_term = num / den
    free_carrier_term = w_p**2 / (omega**2 + 1j * gamma_p * omega)
    return eps_inf * (phonon_term - free_carrier_term)

def eps_lopc(omega, eps_inf, wT, wL, GT, GL, w_p, gamma_p, coupling_strength=0.1):
    eps_base = eps_mdf(omega, eps_inf, wT, wL, GT, GL, w_p, gamma_p)
    omega = np.asarray(omega, dtype=np.complex128)
    coupling_term = coupling_strength * w_p**2 / ((omega - wL)**2 + 1j * GL * omega)
    return eps_base + eps_inf * coupling_term

def reflectance_bulk(eps):
    n = np.sqrt(eps + 0j)
    r = (1 - n) / (1 + n)
    return np.abs(r) ** 2

def _cos_theta_complex(n_in, n_out, theta_in_rad):
    sin_in = np.sin(theta_in_rad)
    ratio = (n_in / n_out) * sin_in
    return np.sqrt(1 - ratio * ratio + 0j)

def reflectance_film_oblique(eps_f, eps_s, d_m, lam_m, theta0_deg=16.0, include_half_wave=False):
    """三层斜入射，未偏振 R=(Rs+Rp)/2；默认不统一 +π，避免相位重复"""
    theta0 = np.deg2rad(theta0_deg)
    n0 = 1.0 + 0j
    n1 = np.sqrt(eps_f + 0j)
    n2 = np.sqrt(eps_s + 0j)

    cos0 = np.cos(theta0)
    cos1 = _cos_theta_complex(n0, n1, theta0)
    cos2 = _cos_theta_complex(n0, n2, theta0)

    r01_s = (n0 * cos0 - n1 * cos1) / (n0 * cos0 + n1 * cos1)
    r12_s = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
    r01_p = (n1 * cos0 - n0 * cos1) / (n1 * cos0 + n0 * cos1)
    r12_p = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)

    k0 = 2.0 * np.pi / lam_m
    delta = k0 * n1 * d_m * cos1
    if include_half_wave and abs(theta0_deg) < 1e-9:
        delta = delta + np.pi

    e2i = np.exp(2j * delta)
    rs = (r01_s + r12_s * e2i) / (1.0 + r01_s * r12_s * e2i)
    rp = (r01_p + r12_p * e2i) / (1.0 + r01_p * r12_p * e2i)
    Rs = np.abs(rs) ** 2
    Rp = np.abs(rp) ** 2
    return 0.5 * (Rs + Rp)

def reflectance_film(eps_film, eps_sub, d_m, lambda_m):
    return reflectance_film_oblique(eps_film, eps_sub, d_m, lambda_m,
                                    theta0_deg=0.0, include_half_wave=False)

def plasma_omega_from_N(N_cm3, eps_inf, m_eff_SI):
    N_m3 = N_cm3 * 1e6
    return math.sqrt(N_m3 * e_charge * e_charge / (eps0 * eps_inf * m_eff_SI))

def gamma_p_from_mobility(mu_cm2_Vs, m_eff_mob_SI):
    mu_m2_Vs = mu_cm2_Vs * 1e-4
    return e_charge / (m_eff_mob_SI * mu_m2_Vs)

def GL_from_concentration(log10_N, polytype="4H"):
    if polytype == "4H": GL_est = 2.0 + 0.8 * (log10_N - 17.0)
    else:                GL_est = 2.0 + 0.7 * (log10_N - 17.0)
    return np.clip(GL_est, 1.0, 15.0)

def eps_from_params_advanced(wnum_cm, eps_inf, wT_cm, wL_cm, GT_cm, GL_cm, N_cm3, mu_cm2_Vs,
                             mconc_SI, mmob_SI, use_lopc=False):
    w = wnum_cm_to_omega(wnum_cm)
    wT = wnum_cm_to_omega(wT_cm)
    wL = wnum_cm_to_omega(wL_cm)
    GT = wnum_cm_to_omega(GT_cm)
    GL = wnum_cm_to_omega(GL_cm)
    wp = plasma_omega_from_N(N_cm3, eps_inf, mconc_SI)
    gp = gamma_p_from_mobility(mu_cm2_Vs, mmob_SI)
    if use_lopc and N_cm3 > 1e19:
        return eps_lopc(w, eps_inf, wT, wL, GT, GL, wp, gp)
    else:
        return eps_mdf(w, eps_inf, wT, wL, GT, GL, wp, gp)

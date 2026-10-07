"""Original Q2 fitting API; see docs/methods.md for cost/initialisation caveats."""
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares, differential_evolution
from .forward_model import (m_eff_4H_concentration,m_eff_4H_mobility,m_eff_6H_isotropic,
    wnum_cm_to_lambda_m,eps_from_params_advanced,reflectance_film_oblique,
    GL_from_concentration,c_cm_per_s,eps0,e_charge,plasma_omega_from_N,gamma_p_from_mobility)

def frequency_weights(wnum_cm):
    w = np.ones_like(wnum_cm)
    w[(wnum_cm >= 700) & (wnum_cm <= 1100)] *= 2.0
    w[(wnum_cm >= 300) & (wnum_cm <= 700)]  *= 1.5
    return w

def select_extrema_for_fod(wnum_cm, R, wmin=1700, wmax=3000):
    mask = (wnum_cm >= wmin) & (wnum_cm <= wmax)
    w, r = wnum_cm[mask], R[mask]
    if len(w) < 10: return np.array([]), np.array([]), np.array([])
    prominence = 0.2 * np.std(r)
    pk, _ = find_peaks(r, prominence=prominence, distance=10)
    vk, _ = find_peaks(-r, prominence=prominence, distance=10)
    if len(pk) == 0 and len(vk) == 0: return np.array([]), np.array([]), np.array([])
    idx = np.sort(np.concatenate([pk, vk]))
    return w[idx], r[idx], idx

def fod_initial_thickness(wnum_cm, R, eps_func, theta0_deg=16.0):
    w_sel, r_sel, _ = select_extrema_for_fod(wnum_cm, R)
    if len(w_sel) < 5: return None
    try:
        ref_i = np.argmax(w_sel)
        order = np.arange(len(w_sel))
        K = np.abs(order - ref_i)
        eps_f = eps_func(w_sel)
        n_f = np.sqrt(eps_f + 0j)
        theta0 = np.deg2rad(theta0_deg)
        cos1 = np.sqrt(1 - (np.sin(theta0)**2) / (n_f**2) + 0j)
        omega = w_sel
        X = np.abs(omega / ((n_f * cos1).real + 1e-12))
        Y = 0.5 * K + 0.5
        if len(X) >= 3:
            A_num = np.sum(X * Y) - np.sum(X) * np.sum(Y) / len(X)
            A_den = np.sum(X * X) - np.sum(X)**2 / len(X)
            if abs(A_den) > 1e-12:
                A = A_num / A_den
                d_cm = abs(A) / 2.0 * 1e-4
                d_um = d_cm * 1e4
                d_um = np.clip(d_um, 0.5, 30.0)
                return float(d_um), float(np.mean(Y) - A * np.mean(X))
        return None
    except Exception:
        return None

def estimate_initial_params_enhanced(wnum_cm, R_meas, use_fod=True, theta0_deg=16.0):
    R_smooth = np.convolve(R_meas, np.ones(21)/21, mode='same')
    R_detrend = R_meas - R_smooth
    peaks, _ = find_peaks(R_detrend, height=0.01)
    valleys, _ = find_peaks(-R_detrend, height=0.01)
    if len(peaks) > 1 and len(valleys) > 1:
        peak_spacing = np.mean(np.diff(wnum_cm[peaks]))
        valley_spacing = np.mean(np.diff(wnum_cm[valleys]))
        fringe_spacing = (peak_spacing + valley_spacing) / 2
        n_approx = 2.6
        theta0_rad = np.deg2rad(theta0_deg)
        cos_theta1_approx = np.sqrt(1 - (np.sin(theta0_rad) / n_approx)**2)
        d_um_est = 10000 / (2 * n_approx * cos_theta1_approx * fringe_spacing)
        d_um_est = np.clip(d_um_est, 1.0, 25.0)
    else:
        d_um_est = 7.0
    if use_fod:
        def temp_eps_func(wn): return 6.56 * np.ones_like(wn, dtype=complex)
        ret = fod_initial_thickness(wnum_cm, R_meas, temp_eps_func, theta0_deg)
        if ret is not None and 0.5 <= ret[0] <= 30.0:
            d_um_est = ret[0]
    # 掺杂/阻尼初估
    rest = (wnum_cm > 700) & (wnum_cm < 1100)
    if np.any(rest):
        R_rest = R_meas[rest]; w_rest = wnum_cm[rest]
        min_idx = np.argmin(R_rest); plasma_cm = w_rest[min_idx]
        eps_inf = 6.56; meff = m_eff_4H_concentration()
        omega_p = 2 * np.pi * c_cm_per_s * plasma_cm
        N_est = (omega_p**2 * eps0 * eps_inf * meff) / (e_charge**2 * 1e6)
        N_est = np.clip(N_est, 1e16, 1e19)
        log10_N_f_est = np.log10(N_est)
    else:
        log10_N_f_est = 17.5
    mu_f_est, mu_s_est = 400.0, 80.0
    log10_N_s_est = log10_N_f_est + 0.3
    GT_f_est, GT_s_est = 2.5, 3.0
    GL_f_est = GL_from_concentration(log10_N_f_est)
    GL_s_est = GL_from_concentration(log10_N_s_est)
    return dict(log10_N_f=log10_N_f_est, mu_f=mu_f_est, d_um=d_um_est,
                log10_N_s=log10_N_s_est, mu_s=mu_s_est,
                GT_f_cm=GT_f_est, GT_s_cm=GT_s_est,
                GL_f_cm=GL_f_est, GL_s_cm=GL_s_est)

def check_high_doping_warning(log10_N_f, log10_N_s):
    out = []
    if log10_N_f > 19.0: out.append("外延层浓度 > 1e19 cm⁻³，建议启用 LOPC")
    if log10_N_s > 19.0: out.append("衬底浓度 > 1e19 cm⁻³，可能需要 LOPC 修正")
    return out

def fit_stack_enhanced(wnum_cm, R_meas, polytype="4H", method="hybrid", use_weights=True,
                      theta0_deg=16.0, include_half_wave=False, use_fod_init=True,
                      use_lopc=False, joint_fod_weight=0.0, x0_override=None,
                      tail_emphasis=1.0):
    if polytype == "4H":
        eps_inf, wT_cm, wL_cm = 6.56, 798.0, 970.0
        mconc_SI, mmob_SI = m_eff_4H_concentration(), m_eff_4H_mobility()
    else:
        eps_inf, wT_cm, wL_cm = 6.52, 797.0, 969.4
        mconc_SI = mmob_SI = m_eff_6H_isotropic()

    init_est = estimate_initial_params_enhanced(wnum_cm, R_meas, use_fod_init, theta0_deg)
    print("初值：d=%.2f μm, log10N_f=%.2f, log10N_s=%.2f, θ=%.1f°"
          % (init_est['d_um'], init_est['log10_N_f'], init_est['log10_N_s'], theta0_deg))
    warn = check_high_doping_warning(init_est['log10_N_f'], init_est['log10_N_s'])
    for w in warn: print("  ⚠", w)
    if not use_lopc and (init_est['log10_N_f'] > 19.0 or init_est['log10_N_s'] > 19.0):
        print("  自动启用 LOPC"); use_lopc = True

    # 边界（d ∈ [5, 10] μm）
    lb = np.array([15.0, 10,   5.0,  16.0, 10,  1.0, 1.0, 1.0, 1.0])
    ub = np.array([19.8, 800, 10.0, 19.8, 500, 4.0, 4.0, 15.0, 15.0])
    x0 = np.array([
        init_est["log10_N_f"], init_est["mu_f"], init_est["d_um"],
        init_est["log10_N_s"], init_est["mu_s"],
        init_est["GT_f_cm"],   init_est["GT_s_cm"],
        init_est["GL_f_cm"],   init_est["GL_s_cm"]
    ], dtype=float)
    x0 = np.clip(x0, lb, ub)
    if x0_override is not None:
        x0 = np.clip(np.asarray(x0_override, dtype=float), lb, ub)

    lam = wnum_cm_to_lambda_m(wnum_cm)
    if use_weights:
        weights = frequency_weights(wnum_cm)
        if tail_emphasis > 1.0:
            weights[wnum_cm >= 1000] *= tail_emphasis
        weights = np.sqrt(weights)
    else:
        weights = np.ones_like(wnum_cm)

    def model(theta):
        logNf, muf, d_um, logNs, mus, GTf, GTs, GLf, GLs = theta
        eps_f = eps_from_params_advanced(wnum_cm, eps_inf, wT_cm, wL_cm, GTf, GLf,
                                         10**logNf, muf, mconc_SI, mmob_SI, use_lopc)
        eps_s = eps_from_params_advanced(wnum_cm, eps_inf, wT_cm, wL_cm, GTs, GLs,
                                         10**logNs, mus, mconc_SI, mmob_SI, use_lopc)
        return reflectance_film_oblique(eps_f, eps_s, d_um * 1e-6, lam, theta0_deg, include_half_wave)

    def resid(theta): return (model(theta) - R_meas) * weights

    best_result, best_cost = None, np.inf
    if method in ["hybrid", "leastsq"]:
        print("尝试最小二乘...")
        try:
            res1 = least_squares(resid, x0, bounds=(lb, ub),
                                 loss="soft_l1", f_scale=0.01,
                                 xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=8000)
            if res1.cost < best_cost:
                best_result, best_cost = res1, res1.cost
                print("  LS cost: %.4e" % res1.cost)
        except Exception as e:
            print("  最小二乘失败：", e)

    if method in ["hybrid", "differential"]:
        print("尝试差分进化（全局）...")
        try:
            bounds_de = list(zip(lb, ub))
            def objective(theta): return np.sum(resid(theta)**2)
            res2 = differential_evolution(objective, bounds_de, seed=42,
                                          maxiter=500, popsize=15, atol=1e-8, tol=1e-8)
            if res2.fun < best_cost:
                best_result = type('Result', (), {'x': res2.x, 'cost': res2.fun, 'success': res2.success})()
                best_cost = res2.fun
                print("  DE cost: %.4e" % res2.fun)
        except Exception as e:
            print("  差分进化失败：", e)

    if best_result is None:
        print("优化均失败，回退初值")
        best_result = type('Result', (), {'x': x0, 'cost': float(np.sum(resid(x0)**2)), 'success': False})()

    theta = best_result.x
    R_fit = model(theta)

    out = {
        "success": bool(getattr(best_result, 'success', False)),
        "cost": float(best_cost),
        "rms_error": float(np.sqrt(np.mean((R_fit - R_meas)**2))),
        "theta": {
            "log10_N_film": float(theta[0]), "mu_film_cm2_Vs": float(theta[1]), "d_um": float(theta[2]),
            "log10_N_sub":  float(theta[3]), "mu_sub_cm2_Vs":  float(theta[4]),
            "GT_film_cm^-1": float(theta[5]), "GT_sub_cm^-1": float(theta[6]),
            "GL_film_cm^-1": float(theta[7]), "GL_sub_cm^-1": float(theta[8])
        },
        "derived": {},
        "wnum_cm": wnum_cm.tolist(),
        "R_meas": R_meas.tolist(),
        "R_fit": R_fit.tolist(),
        "meta": {
            "polytype": polytype, "eps_inf": eps_inf,
            "wT_cm": wT_cm, "wL_cm": wL_cm,
            "mconc_SI": mconc_SI, "mmob_SI": mmob_SI,
            "theta0_deg": theta0_deg, "include_half_wave": include_half_wave,
            "use_lopc": use_lopc, "joint_fod_weight": joint_fod_weight,
            "warnings": warn
        }
    }

    for layer, logN, mu, GT, GL in [
        ("film", theta[0], theta[1], theta[5], theta[7]),
        ("substrate", theta[3], theta[4], theta[6], theta[8])
    ]:
        N_cm3 = 10.0 ** logN
        wp = plasma_omega_from_N(N_cm3, eps_inf, mconc_SI)
        gp = gamma_p_from_mobility(mu, mmob_SI)
        out["derived"][layer] = {
            "N_cm^-3": float(N_cm3),
            "mu_cm2_Vs": float(mu),
            "GT_cm^-1": float(GT),
            "GL_cm^-1": float(GL),
            "omega_p_rad_s": float(wp),
            "gamma_p_rad_s": float(gp),
        }

    return out

def _pack_theta(theta):
    return np.array([
        theta['log10_N_film'], theta['mu_film_cm2_Vs'], theta['d_um'],
        theta['log10_N_sub'],  theta['mu_sub_cm2_Vs'],
        theta['GT_film_cm^-1'],theta['GT_sub_cm^-1'],
        theta['GL_film_cm^-1'],theta['GL_sub_cm^-1']
    ], dtype=float)

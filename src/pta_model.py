"""Analytical model matching for build-ups (what commercial PTA software does).

Instead of a straight line through one part of the curve (Horner), a full reservoir model is fitted to the
WHOLE build-up. Model = radial flow to a vertical well with:
  - wellbore storage C [bbl/psi]: after the surface valve closes, the reservoir keeps feeding the wellbore
    (fluid compresses in it), so the sandface rate drops to 0 gradually -> early unit-slope + hump
  - skin S: extra pressure drop near the well
  - outer boundary: 'infinite' | 'constant_pressure' circle (aquifer / injector support) | 'closed' circle

Dimensionless variables (field units, t in hours):
  tD = 0.0002637 k t / (phi mu ct rw^2)           time
  CD = 0.8936 C / (phi ct h rw^2)                 wellbore storage
  dP = 141.2 q B mu / (k h) * pD                  pressure
  reD = re / rw                                   boundary radius

The flow equation is solved exactly in the Laplace domain (variable s) and converted back to time with the
Stehfest algorithm (a standard numerical inverse Laplace transform).
Build-up via superposition (flow at q for tp, then shut in):
  dP_buildup(dt) = Pws - Pwf = 141.2 q B mu / (k h) * [pD(dt) - pD(tp + dt) + pD(tp)]
"""
import numpy as np
from math import factorial
from scipy.optimize import least_squares
from scipy.special import i0e, i1e, k0e, k1e

BOUNDARIES = ['infinite', 'constant_pressure', 'closed']


# ---------------------------------------------------------------------------------------------
# Laplace-domain solution + Stehfest inversion
# ---------------------------------------------------------------------------------------------
def stehfest_weights(n=12):
    """Stehfest (1970) weights V_i. p(t) = ln2/t * sum_i V_i * p_bar(i * ln2 / t)."""
    v = np.zeros(n)
    half = n // 2
    for i in range(1, n + 1):
        total = 0.0
        for k in range((i + 1) // 2, min(i, half) + 1):
            total += (k ** half * factorial(2 * k) /
                      (factorial(half - k) * factorial(k) * factorial(k - 1) *
                       factorial(i - k) * factorial(2 * k - i)))
        v[i - 1] = (-1) ** (i + half) * total
    return v


WEIGHTS = stehfest_weights()


def pwd_laplace(s, CD, S, reD, boundary):
    """Wellbore pressure in Laplace space.

    Reservoir (sandface) solution, a = sqrt(s), b = reD * sqrt(s)  (Van Everdingen & Hurst):
      infinite          : K0(a) / (s a K1(a))
      constant pressure : [K0(a) I0(b) - I0(a) K0(b)] / (s a [K1(a) I0(b) + I1(a) K0(b)])
      closed            : [K0(a) I1(b) + I0(a) K1(b)] / (s a [I1(b) K1(a) - I1(a) K1(b)])
    Exponentially-scaled Bessel functions (i0e, k0e, ...) are used so large arguments don't overflow;
    the scale factors cancel between numerator and denominator.
    Then skin and storage:   p0 = p_res + S / s,     p_w = p0 / (1 + CD s^2 p0)
    """
    a = np.sqrt(s)
    if boundary == 'infinite':
        p_res = k0e(a) / (s * a * k1e(a))
    else:
        b = reD * a
        e = np.exp(2 * a - 2 * b)
        if boundary == 'constant_pressure':
            num = k0e(a) * i0e(b) - i0e(a) * k0e(b) * e
            den = k1e(a) * i0e(b) + i1e(a) * k0e(b) * e
        else:
            num = k0e(a) * i1e(b) + i0e(a) * k1e(b) * e
            den = i1e(b) * k1e(a) - i1e(a) * k1e(b) * e
        p_res = num / (s * a * den)
    p0 = p_res + S / s
    return p0 / (1 + CD * s ** 2 * p0)


def pwd(tD, CD, S, reD, boundary):
    """Dimensionless wellbore pressure vs dimensionless time (Stehfest inversion)."""
    tD = np.atleast_1d(tD).astype(float)
    i = np.arange(1, len(WEIGHTS) + 1)
    s = np.outer(np.log(2) / tD, i)                       # one row of s-values per time
    return np.log(2) / tD * (pwd_laplace(s, CD, S, reD, boundary) @ WEIGHTS)


# ---------------------------------------------------------------------------------------------
# Build-up model in field units
# ---------------------------------------------------------------------------------------------
def buildup_dp(dt_hr, k, S, C, re_ft, q, tp, pvt, boundary):
    """Model dP = Pws - Pwf [psi] for a build-up after producing q [STB/d] for tp [h]."""
    phi, mu, ct = pvt['porosity_frac'], pvt['oil_viscosity_cp'], pvt['total_compressibility_1_psi']
    h, rw, B = pvt['net_pay_thickness_ft'], pvt['wellbore_radius_ft'], pvt['oil_fvf_rb_stb']
    td_per_hr = 0.0002637 * k / (phi * mu * ct * rw ** 2)
    CD = 0.8936 * C / (phi * ct * h * rw ** 2)
    reD = re_ft / rw
    p = lambda t: pwd(t * td_per_hr, CD, S, reD, boundary)
    pd_buildup = p(dt_hr) - p(tp + dt_hr) + p(np.array([tp]))[0]
    return 141.2 * q * B * mu / (k * h) * pd_buildup


def fit_model(dt, dp, q, tp, pvt, boundary, k0, s0):
    """Least-squares fit of the model to the log-binned build-up (errors in log(dP) = relative errors,
    so early and late times count equally). Parameters searched in log space where natural.
    Start values: k, s from Horner; C from the early unit-slope (dP = q B dt / (24 C)); re from drainage area.
    """
    C0 = q * pvt['oil_fvf_rb_stb'] * dt[0] / (24 * dp[0])
    re0 = np.sqrt(pvt['estimated_drainage_area_acres'] * 43560 / np.pi)

    def unpack(x):
        """x = [log10 k, S, log10 C, (log10 re)]; the infinite model has no re."""
        re = 10 ** x[3] if len(x) == 4 else np.nan
        return 10 ** x[0], x[1], 10 ** x[2], re

    def residuals(x):
        k, S, C, re = unpack(x)
        model = buildup_dp(dt, k, S, C, re, q, tp, pvt, boundary)
        if np.any(~np.isfinite(model)) or np.any(model <= 0):
            return np.full(len(dt), 10.0)
        return np.log(model) - np.log(dp)

    x0 = [np.log10(k0), s0, np.log10(C0), np.log10(re0)]
    lower = [-1, -6, -5, np.log10(50)]
    upper = [4, 50, 1, np.log10(50000)]
    if boundary == 'infinite':                           # no boundary radius to fit
        x0, lower, upper = x0[:3], lower[:3], upper[:3]
    fit = least_squares(residuals, x0, bounds=(lower, upper))
    k, S, C, re = unpack(fit.x)
    rms_pct = np.sqrt(np.mean(fit.fun ** 2)) * 100
    n, p = len(dt), len(fit.x)
    aic = n * np.log(np.mean(fit.fun ** 2)) + 2 * p     # model comparison: lower = better (penalises extra parameters)
    return {'boundary': boundary, 'k_md': k, 'skin': S, 'C_bbl_psi': C,
            'boundary_radius_ft': re,
            'rms_error_pct': rms_pct, 'aic': aic,
            '_model_dp': buildup_dp(dt, k, S, C, re, q, tp, pvt, boundary)}


def radius_of_investigation(k, t_hr, pvt):
    """r_i = sqrt(k t / (948 phi mu ct)) [ft]: how far the pressure wave travelled in t hours."""
    return np.sqrt(k * t_hr / (948 * pvt['porosity_frac'] * pvt['oil_viscosity_cp'] *
                               pvt['total_compressibility_1_psi']))


def fit_all_models(result, pvt, min_aic_gain=4):
    """Fit all three boundary models to one analysed event (output of pta.analyze_event) and choose one.

    A boundary model has one extra parameter, so it nearly always fits a little better. It is accepted only if
      1. AIC improves by more than `min_aic_gain` vs the infinite model (clear improvement, not just the
         extra parameter), AND
      2. the fitted boundary lies inside the radius of investigation (the pressure wave reached it during
         the test -- otherwise the boundary cannot have been 'seen').
    Otherwise the simpler infinite-acting model is kept.
    """
    dt, dp = result['_dt_bins'], result['_dp_bins']
    keep = dp > 0
    dt, dp = dt[keep], dp[keep]
    fits = [fit_model(dt, dp, result['q_oil_stbd'], result['tp_hr'], pvt, b, result['k_md'], result['skin'])
            for b in BOUNDARIES]

    infinite = fits[0]
    best = infinite
    for f in fits[1:]:
        r_inv = radius_of_investigation(f['k_md'], dt[-1], pvt)
        f['radius_investigation_ft'] = r_inv
        seen = f['boundary_radius_ft'] < r_inv
        if seen and infinite['aic'] - f['aic'] > min_aic_gain and f['aic'] < best['aic']:
            best = f
    infinite['radius_investigation_ft'] = radius_of_investigation(infinite['k_md'], dt[-1], pvt)
    return fits, best, dt, dp


def plot_model_match(result, fits, best, dt, dp):
    """Log-log plot: measured dP and derivative (dots) vs each model (lines). Best model in bold."""
    import matplotlib.pyplot as plt
    from pta import bourdet_derivative
    tp = result['tp_hr']
    fig, ax = plt.subplots(figsize=(8, 5.5))
    der = bourdet_derivative(dt, dp, tp)
    ax.loglog(dt, dp, 'o', ms=3, color='tab:blue', label='dP measured')
    ax.loglog(dt[der > 0], der[der > 0], 's', ms=3, color='tab:red', label="dP' measured")
    styles = {'infinite': ':', 'constant_pressure': '--', 'closed': '-.'}
    for f in fits:
        lw = 2.5 if f is best else 1
        mder = bourdet_derivative(dt, f['_model_dp'], tp)
        label = f"{f['boundary']} (rms {f['rms_error_pct']:.2f} %)" + ('  <- chosen' if f is best else '')
        ax.loglog(dt, f['_model_dp'], styles[f['boundary']], color='k', lw=lw, label=label)
        ax.loglog(dt[mder > 0], mder[mder > 0], styles[f['boundary']], color='k', lw=lw)
    ax.set_xlabel('dt [h]')
    ax.set_ylabel("dP, dP' [psi]")
    re_txt = '' if np.isnan(best['boundary_radius_ft']) else f", re = {best['boundary_radius_ft']:.0f} ft"
    ax.set_title(f"{result['well_name']} {result['t_shut']:%Y-%m-%d}: model match -> k = {best['k_md']:.1f} md, "
                 f"S = {best['skin']:.2f}, C = {best['C_bbl_psi']:.3f} bbl/psi{re_txt}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

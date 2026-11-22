"""Synthetic build-up benchmark: simulated shut-ins with KNOWN k, skin, slope, dP skin.

Why: without a reference answer, the rigorous way to compare PTA methods is to simulate realistic build-ups
where the truth is known, run every method on them, and measure the errors. Realism:
  - the analytical reservoir model of pta_model.py (wellbore storage + skin + boundary, checked against
    textbook limits),
  - each well's real PVT and real shut-in duration,
  - the REAL variable rate history before the shut-in (VFM daily rates), via superposition:
        Pw(t) = Pi - 141.2 B mu / (k h) * sum_i (q_i - q_(i-1)) * pD(t - t_i)
  - gauge noise.
Truth for the four scored parameters:
  m_true = 162.6 q B mu / (k h)  (q = rate just before shut-in),  dP_skin_true = 0.869 m_true s
"""
import numpy as np
import pandas as pd

import pta_model as M

HISTORY_YEARS = 5           # constant-rate production before 2025 (the wells are 5-15 years old)


def rate_steps(daily_oil, t_shut):
    """Rate history as (start time [h, relative to shut-in], rate) steps, oldest first.
    Last 30 days: daily rates; earlier 2025: weekly averages; before 2025: one long constant period."""
    d = daily_oil[daily_oil.date < t_shut.normalize()].sort_values('date')
    q_shut_day = d.oil.iloc[-1]                                  # the day before -> flowing rate on shut-in day
    recent = d[d.date >= t_shut.normalize() - pd.Timedelta(days=30)]
    older = d[d.date < t_shut.normalize() - pd.Timedelta(days=30)]
    steps = [(-(t_shut - pd.Timestamp('2025-01-01')).total_seconds() / 3600 - HISTORY_YEARS * 8760,
              d.oil.iloc[:30].mean())]
    if len(older):
        weekly = older.set_index('date').oil.resample('7D').mean().dropna()
        steps += [((t - t_shut).total_seconds() / 3600, q) for t, q in weekly.items()]
    steps += [((t - t_shut).total_seconds() / 3600, q) for t, q in zip(recent.date, recent.oil)]
    steps += [((t_shut.normalize() - t_shut).total_seconds() / 3600, q_shut_day), (0.0, 0.0)]
    return steps, q_shut_day


def simulate(steps, pvt, k, skin, C, boundary, re_ft, duration_hr, noise_psi=0.3, p_init=4000.0, seed=0):
    """Minute pressures of one simulated build-up. Returns dt [h], pws [psia], pwf [psia] (just before shut-in)."""
    rng = np.random.default_rng(seed)
    phi, mu, ct = pvt['porosity_frac'], pvt['oil_viscosity_cp'], pvt['total_compressibility_1_psi']
    h, rw, B = pvt['net_pay_thickness_ft'], pvt['wellbore_radius_ft'], pvt['oil_fvf_rb_stb']
    td_per_hr = 0.0002637 * k / (phi * mu * ct * rw ** 2)
    CD = 0.8936 * C / (phi * ct * h * rw ** 2)
    reD = re_ft / rw
    scale = 141.2 * B * mu / (k * h)

    dt = np.arange(1, int(duration_hr * 60) + 1) / 60.0
    t = np.concatenate([[-1e-4], dt])                            # first point = just before shut-in
    pressure = np.full(len(t), p_init)
    q_prev = 0.0
    for t_i, q_i in steps:
        elapsed = t - t_i
        live = elapsed > 0
        if q_i != q_prev and live.any():
            pressure[live] -= scale * (q_i - q_prev) * M.pwd(elapsed[live] * td_per_hr, CD, skin, reD, boundary)
        q_prev = q_i
    pressure += rng.normal(0, noise_psi, len(pressure))
    return dt, pressure[1:], pressure[0]


def truth(q, pvt, k, skin):
    m = 162.6 * q * pvt['oil_fvf_rb_stb'] * pvt['oil_viscosity_cp'] / (k * pvt['net_pay_thickness_ft'])
    return {'k_md': k, 'skin': skin, 'm_psi_cycle': m, 'dP_skin_psi': 0.869 * m * skin}


def sample_scenarios(wells, n_per_well=20, seed=0):
    """Random but realistic true parameters per well (around that well's own estimate)."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in wells:
        for i in range(n_per_well):
            boundary = rng.choice(['infinite', 'constant_pressure', 'closed'], p=[0.5, 0.3, 0.2])
            rows.append({'well_name': w, 'scenario': i,
                         'k_factor': float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))),
                         'skin': float(rng.uniform(-2, 6)),
                         'C': float(np.exp(rng.uniform(np.log(0.005), np.log(0.05)))),
                         'boundary': boundary,
                         're_ft': (float(rng.uniform(400, 2500)) if boundary == 'constant_pressure'
                                   else float(rng.uniform(1500, 3000)) if boundary == 'closed' else np.nan),
                         'seed': int(rng.integers(1e9))})
    return pd.DataFrame(rows)

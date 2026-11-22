"""Rigorous PTA method comparison on simulated build-ups with KNOWN truth (src/pta_synthetic.py).

For each of the 6 real shut-ins (real well PVT, real duration, real VFM rate history) N random true reservoirs
are simulated; every method analyses every build-up; errors on the four scored parameters are recorded.

Usage (project root):  python src/run_pta_benchmark.py [n_per_well]
Writes outputs/pta_benchmark_results.csv (one row per simulation x method).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import load
import pta
import pta_model
import pta_synthetic as S

START = pd.Timestamp('2025-01-01')


def horner_custom(dt, pws, pwf, q, tp, pvt, window=None, p1hr_measured=False):
    dt_b, p_b = pta.log_bins(dt, pws)
    der = pta.bourdet_derivative(dt_b, p_b - pwf, tp)
    window = window or pta.find_radial_window(dt_b, der)
    m, _, p1hr = pta.fit_horner(dt, pws, tp, window)
    if p1hr_measured:
        p1hr = np.interp(1.0, dt, pws)
    k, skin, dps = pta.reservoir_properties(m, p1hr, pwf, q, pvt)
    return {'k_md': k, 'skin': skin, 'm_psi_cycle': m, 'dP_skin_psi': dps}


def method_results(dt, pws, pwf, ctx):
    """name -> {k_md, skin, m_psi_cycle, dP_skin_psi} for every candidate method."""
    q, cum, pvt, steps = ctx['q'], ctx['cum'], ctx['pvt'], ctx['steps']
    tp = pta.producing_time(cum, q)
    out = {}
    out['Horner auto (v5)'] = horner_custom(dt, pws, pwf, q, tp, pvt)
    out['Horner tp since 2025 (C)'] = horner_custom(dt, pws, pwf, q, (ctx['t_shut'] - START).total_seconds() / 3600, pvt)
    out['Horner window 1h-end (D)'] = horner_custom(dt, pws, pwf, q, tp, pvt, window=(1.0, dt.max()))
    out['Horner P1hr measured (E)'] = horner_custom(dt, pws, pwf, q, tp, pvt, p1hr_measured=True)
    hours = pd.DataFrame({'h': np.floor(dt), 'dt': dt, 'p': pws}).groupby('h').mean()
    out['Horner hourly data (F)'] = horner_custom(hours.dt.values, hours.p.values, pwf, q, tp, pvt)

    # superposition (multi-rate) Horner: same automatic window, line vs superposition time
    dt_b, p_b = pta.log_bins(dt, pws)
    window = pta.find_radial_window(dt_b, pta.bourdet_derivative(dt_b, p_b - pwf, tp))
    m, _, p1hr = pta.fit_superposition(dt, pws, steps, q, window)
    k, skin, dps = pta.reservoir_properties(m, p1hr, pwf, q, pvt)
    out['Superposition Horner'] = {'k_md': k, 'skin': skin, 'm_psi_cycle': m, 'dP_skin_psi': dps}

    # analytical model matching (k, skin from the best model; m and dP skin derived consistently)
    base = pta.analyze_buildup(dt, pws, pwf, q, cum, pvt)
    fits, best, _, _ = pta_model.fit_all_models(base, pvt)
    for label, f in [('Model match (best AIC)', best), ('Model match (infinite)', fits[0])]:
        mm = 162.6 * q * pvt['oil_fvf_rb_stb'] * pvt['oil_viscosity_cp'] / (f['k_md'] * pvt['net_pay_thickness_ft'])
        out[label] = {'k_md': f['k_md'], 'skin': f['skin'], 'm_psi_cycle': mm, 'dP_skin_psi': 0.869 * mm * f['skin']}
    return out


def run(n_per_well=20):
    hf = load.load_high_freq_clean()
    pvt = load.load_pvt().set_index('well_name')
    completion = load.load_completion().set_index('well_name')
    v = pd.read_csv(load.ROOT / 'outputs' / 'submission' / 'vfm_submission_v5.csv')
    rates = pd.DataFrame({'date': pd.to_datetime(v.timestamp), 'well_name': v.well_name, 'oil': v.oil_rate_stbd})
    own = load.load_own_k_skin()
    events = pta.find_shut_ins(hf).set_index('well_name')

    rows = []
    for _, sc in S.sample_scenarios(events.index, n_per_well).iterrows():
        e = events.loc[sc.well_name]
        p = pvt.loc[sc.well_name]
        r = rates[rates.well_name == sc.well_name]
        steps, q = S.rate_steps(r, e.t_shut)
        cum = completion.loc[sc.well_name, 'cum_oil_at_2025_01_01_stb'] + r[r.date < e.t_shut.normalize()].oil.sum()
        k_true = own.loc[sc.well_name, 'k_md'] * sc.k_factor
        dt, pws, pwf = S.simulate(steps, p, k_true, sc.skin, sc.C, sc.boundary, sc.re_ft, e.duration_hr, seed=sc.seed)
        truth = S.truth(q, p, k_true, sc.skin)
        ctx = {'q': q, 'cum': cum, 'pvt': p, 'steps': steps, 't_shut': e.t_shut}
        try:
            results = method_results(dt, pws, pwf, ctx)
        except Exception as exc:                                    # keep going, record the failure
            print('failed', sc.well_name, sc.scenario, exc, flush=True)
            continue
        for method, est in results.items():
            rows.append({'well_name': sc.well_name, 'scenario': sc.scenario, 'boundary': sc.boundary,
                         'duration_hr': e.duration_hr, 'method': method,
                         **{f'true_{k}': v for k, v in truth.items()}, **{f'est_{k}': v for k, v in est.items()}})
    out = pd.DataFrame(rows)
    out['k_err_pct'] = (out.est_k_md / out.true_k_md - 1) * 100
    out['m_err_pct'] = (out.est_m_psi_cycle / out.true_m_psi_cycle - 1) * 100
    out['skin_err'] = out.est_skin - out.true_skin
    out['dps_err_psi'] = out.est_dP_skin_psi - out.true_dP_skin_psi
    out.to_csv(load.ROOT / 'outputs' / 'pta_benchmark_results.csv', index=False)
    return out


def summary(out):
    g = out.groupby('method')
    return pd.DataFrame({
        'k |err| % (median)': g.k_err_pct.apply(lambda s: s.abs().median()),
        'k |err| % (p90)': g.k_err_pct.apply(lambda s: s.abs().quantile(0.9)),
        'slope |err| % (median)': g.m_err_pct.apply(lambda s: s.abs().median()),
        'skin |err| (median)': g.skin_err.apply(lambda s: s.abs().median()),
        'skin |err| (p90)': g.skin_err.apply(lambda s: s.abs().quantile(0.9)),
        'dP skin |err| psi (median)': g.dps_err_psi.apply(lambda s: s.abs().median()),
        'k bias %': g.k_err_pct.median(), 'skin bias': g.skin_err.median(),
    }).sort_values('k |err| % (median)')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = run(n)
    pd.set_option('display.width', 250)
    print(summary(out).round(3).to_string())

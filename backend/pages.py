"""Data for the VFM page and the Insights page (reads the active results bundle via backend.data).

Engineering logic stays simple and is explained next to each number:
  - trends over the last 90 days are straight-line fits to the VFM daily values
  - stimulation upside uses the flow efficiency from the build-up analysis
  - bubble-point warnings compare reservoir / flowing pressure with Pb from the PVT file
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from backend import data
import load  # noqa: E402  (src/ is on the path via backend.data)

DAYS_PER_MONTH = 30.4


def recent_slope(dates, values, days=90):
    """Straight-line slope of the last `days` days, per month.
    slope = d(value)/d(day) * 30.4   (e.g. psi/month, percentage points/month)
    Intuition: how fast a quantity is moving right now, not averaged over the whole year."""
    d = pd.DataFrame({'date': pd.to_datetime(dates), 'v': values}).dropna()
    d = d[d.date > d.date.max() - pd.Timedelta(days=days)]
    if len(d) < 10:
        return None
    x = (d.date - d.date.min()).dt.total_seconds() / 86400
    return float(np.polyfit(x, d.v, 1)[0] * DAYS_PER_MONTH)


def _well_rates(well):
    r = data.daily_rates()
    return r[r.well_name == well].sort_values('date')


@lru_cache
def _training_k_skin():
    """k and skin per well that the STORED VFM model was calibrated with: our own build-up analysis of the
    training period (Jan-Sep 2025, data/processed/own_pta_k_skin.csv). Its Darcy J is what the model uses, whatever
    dataset is active. Wells without a shut-in in the training data have none -> their J was fitted from tests."""
    ks = load.load_own_k_skin()
    return {} if ks is None else ks.to_dict('index')


def well_summary(well):
    """One row of the VFM well table: latest rates, calibration and trends."""
    g = _well_rates(well)
    flowing = g[g.on_stream_frac > 0]
    last = flowing.iloc[-1] if len(flowing) else g.iloc[-1]
    model = data.vfm_model()
    tr = model['trends'].loc[well] if well in model['trends'].index else None
    pvt = data.bundle()['pvt'].loc[well]
    ks = _training_k_skin().get(well)
    tests = data.bundle()['tests']
    t = tests[tests.well_name == well] if tests is not None else None
    wape = None
    if t is not None and len(t):
        wape = float((t.pred_oil_rate_stbd - t.oil_rate_stbd).abs().sum() / t.oil_rate_stbd.abs().sum() * 100)
    return {
        'well': well,
        'oil': last.oil, 'water': last.water, 'gas': last.gas, 'water_cut': last.water_cut,
        'oil_30d': g[g.date > g.date.max() - pd.Timedelta(days=30)].oil.mean(),
        'cum_oil': g.oil.sum(),
        'p_res': last.p_res, 'pwf': last.pwf,
        'p_res_slope_psi_month': recent_slope(g.date, g.p_res),
        'wc_slope_pp_month': recent_slope(flowing.date, flowing.water_cut),
        'J': tr.J if tr is not None else None,
        'J_source': ('own build-up, Darcy (Jan-Sep)' if ks else 'fitted from well tests') if tr is not None else None,
        'k_md': ks['k_md'] if ks else None, 'skin': ks['skin'] if ks else None,
        'rs_scf_stb': float(tr.gor_coef[0]) if tr is not None else None,
        'pb_psia': pvt.bubble_point_pressure_psia,
        'n_tests': 0 if t is None else len(t), 'oil_wape_tests': wape,
    }


def vfm_overview():
    """Model card + one summary row per well."""
    model = data.vfm_model()
    wells = sorted(data.daily_rates().well_name.unique())
    tests = data.bundle()['tests']
    return {
        'model': {'name': model['name'], 'options': model['physics_options'], 'trained_until': model['trained_until'],
                  'gp_weights': model['gp_weights'], 'gp_features': model['features'], 'wells': model['wells']},
        'validation': data.vfm_validation(),
        'accuracy_on_dataset_tests': data.dataset_info()['accuracy_on_uploaded_tests'],
        'has_tests': tests is not None,
        'wells': [well_summary(w) for w in wells],
    }


def vfm_well(well):
    """Everything the VFM page plots for one well: daily rates, measured tests, reservoir pressure."""
    g = _well_rates(well)
    model = data.vfm_model()
    tests = data.bundle()['tests']
    t = tests[tests.well_name == well].sort_values('time') if tests is not None else None
    out = {
        'summary': well_summary(well),
        'daily': {'date': g.date.dt.strftime('%Y-%m-%d').tolist(), 'oil': g.oil.round(1).tolist(),
                  'water': g.water.round(1).tolist(), 'gas': g.gas.round(1).tolist(),
                  'water_cut': g.water_cut.round(2).tolist(), 'p_res': g.p_res.round(1).tolist(),
                  'pwf': g.pwf.round(1).tolist(), 'choke': g.choke.round(1).tolist(),
                  'gas_lift': g.gas_lift.round(1).tolist(), 'on_stream': g.on_stream_frac.round(3).tolist()},
        'tests': None, 'pres_obs': None,
    }
    if t is not None and len(t):
        out['tests'] = {'time': t.time.dt.strftime('%Y-%m-%d %H:%M').tolist(),
                        **{k: t[c].round(2).tolist() for k, c in [
                            ('oil', 'oil_rate_stbd'), ('water', 'water_rate_stbd'), ('gas', 'gas_rate_mscfd'),
                            ('pred_oil', 'pred_oil_rate_stbd'), ('pred_water', 'pred_water_rate_stbd'),
                            ('pred_gas', 'pred_gas_rate_mscfd')]}}
    if well in model['trends'].index:
        # reservoir pressure implied by each TRAINING test: P_res = Pwf + q_oil / J  (inverting the inflow equation)
        days, p = model['trends'].loc[well, '_pres_obs']
        out['pres_obs'] = {'date': (pd.Timestamp('2025-01-01') + pd.to_timedelta(days, unit='D')).strftime('%Y-%m-%d').tolist(),
                           'p_res': np.round(p, 1).tolist()}
    return out


# ---------------------------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------------------------
def insights(oil_price=70.0):
    """Actions with their value. Every number below comes from the VFM or the build-up analysis."""
    rates = data.daily_rates()
    results = data.pta_results()
    pvt = data.bundle()['pvt']
    wells = sorted(rates.well_name.unique())

    # 1. Stimulation: removing the skin raises the rate to q / FE at the same drawdown.
    #    FE = (P* - Pwf - dP_skin) / (P* - Pwf)  ->  ideal rate q_ideal = q / FE,  upside = q / FE - q   [STB/d]
    stim = []
    for r in results:
        if r['skin'] > 0 and 0 < r['flow_efficiency'] < 1:
            up = r['q_oil_stbd'] / r['flow_efficiency'] - r['q_oil_stbd']
            planned = r['event_type'] == 'planned build-up test'
            stim.append({'well': r['well_name'], 't_shut': r['t_shut'].strftime('%Y-%m-%d'), 'planned': planned,
                         'skin': r['skin'], 'dP_skin_psi': r['dP_skin_psi'], 'flow_efficiency': r['flow_efficiency'],
                         'q_oil_stbd': r['q_oil_stbd'], 'upside_stbd': up, 'value_usd_day': up * oil_price,
                         'value_usd_year': up * oil_price * 365,
                         'confidence': 'high (planned build-up test)' if planned else 'indicative (short unplanned shut-in)'})
    stim.sort(key=lambda s: (-s['planned'], -s['upside_stbd']))

    # 2. Deferred production: oil not produced while shut in (VFM rate before the shut-in x time closed)
    deferred = [{'well': r['well_name'], 't_shut': r['t_shut'].strftime('%Y-%m-%d %H:%M'), 'duration_hr': r['duration_hr'],
                 'event_type': r['event_type'], 'deferred_oil_stb': r['deferred_oil_stb'],
                 'value_usd': r['deferred_oil_stb'] * oil_price} for r in results]

    # 3. Reservoir health per well: depletion speed, distance to the bubble point, water-cut trend
    health = []
    for w in wells:
        g = rates[rates.well_name == w].sort_values('date')
        flowing = g[(g.on_stream_frac > 0.99) & g.pwf.notna()]
        p_res = g.p_res.dropna().iloc[-1] if g.p_res.notna().any() else None
        slope = recent_slope(g.date, g.p_res)
        pb = float(pvt.loc[w, 'bubble_point_pressure_psia'])
        # months until the reservoir pressure reaches Pb at today's decline: (P_res - Pb) / |slope|
        months_to_pb = (p_res - pb) / -slope if (p_res is not None and slope and slope < 0) else None
        pwf_min = flowing[flowing.date > g.date.max() - pd.Timedelta(days=30)].pwf.min()
        wc_now = flowing.water_cut.iloc[-1] if len(flowing) else None
        wc_slope = recent_slope(flowing.date, flowing.water_cut)
        flags = []
        if slope is not None and slope < -15:
            flags.append('fast depletion')
        if pwf_min is not None and pwf_min - pb < 300:
            flags.append('Pwf near bubble point')
        if months_to_pb is not None and months_to_pb < 24:
            flags.append('reservoir reaches Pb within 2 years')
        if wc_slope is not None and wc_slope > 1.0:
            flags.append('water cut rising fast')
        health.append({'well': w, 'p_res': p_res, 'p_res_slope_psi_month': slope, 'pb_psia': pb,
                       'p_res_minus_pb': None if p_res is None else p_res - pb, 'months_to_pb': months_to_pb,
                       'pwf_min_30d': pwf_min, 'pwf_minus_pb': None if pd.isna(pwf_min) else pwf_min - pb,
                       'water_cut': wc_now, 'wc_slope_pp_month': wc_slope, 'flags': flags})

    # 4. Gas-lift efficiency: oil produced per Mscf of injected lift gas (last 30 days, flowing days)
    #    A low ratio = lift gas that could earn more in another well (candidate for a lift-gas optimisation test).
    lift = []
    for w in wells:
        g = rates[(rates.well_name == w) & (rates.date > rates.date.max() - pd.Timedelta(days=30)) & (rates.on_stream_frac > 0.99)]
        if len(g) and g.gas_lift.mean() > 0:
            lift.append({'well': w, 'oil_stbd': g.oil.mean(), 'lift_gas_mscfd': g.gas_lift.mean(),
                         'oil_per_mscf': g.oil.mean() / g.gas_lift.mean(), 'water_cut': g.water_cut.mean()})
    lift.sort(key=lambda x: -x['oil_per_mscf'])
    if lift:
        median = float(np.median([x['oil_per_mscf'] for x in lift]))
        for x in lift:
            x['vs_median_pct'] = (x['oil_per_mscf'] / median - 1) * 100

    # 5. Data gaps: wells with no planned build-up test -> k and skin unknown, J fitted from tests only
    tested = {r['well_name'] for r in results if r['event_type'] == 'planned build-up test'}
    any_shut = {r['well_name'] for r in results}
    last30 = rates[rates.date > rates.date.max() - pd.Timedelta(days=30)].groupby('well_name').oil.mean()
    gaps = [{'well': w, 'oil_stbd': float(last30.get(w, np.nan)),
             'note': 'only a short unplanned shut-in analysed' if w in any_shut else 'no shut-in in the data'}
            for w in wells if w not in tested]
    gaps.sort(key=lambda x: -x['oil_stbd'])

    planned_stim = [s for s in stim if s['planned']]
    return {
        'oil_price': oil_price,
        'kpis': {'stimulation_upside_stbd': sum(s['upside_stbd'] for s in planned_stim),
                 'stimulation_value_usd_year': sum(s['value_usd_year'] for s in planned_stim),
                 'deferred_oil_stb': sum(d['deferred_oil_stb'] for d in deferred),
                 'deferred_value_usd': sum(d['value_usd'] for d in deferred),
                 'wells_flagged': sum(1 for h in health if h['flags']), 'wells_to_test': len(gaps)},
        'stimulation': stim, 'deferred': deferred, 'health': health, 'lift': lift, 'gaps': gaps,
    }

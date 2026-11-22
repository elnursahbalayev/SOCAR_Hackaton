"""Validation views, IPR / what-if and production forecast for the VFM and Well Test Analysis pages.

Static validation evidence of the stored model comes from outputs/validation/validation.json
(tools/build_validation.py); everything dataset-specific is computed from the active results bundle.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from backend import data, pages
import pta  # noqa: E402  (src/ is on the path via backend.data)

VALIDATION_FILE = data.ROOT / 'outputs' / 'validation' / 'validation.json'
PLANNED = 'planned build-up test'


@lru_cache
def stored_validation():
    """Validation evidence of the stored model (it does not change when data is uploaded)."""
    return json.loads(VALIDATION_FILE.read_text(encoding='utf-8'))


def band():
    """Uncertainty band of a daily rate: P10 / P90 of (measured / predicted) in the forecast validation.
    rate_P10 = rate x p10,  rate_P90 = rate x p90.  Forecast errors are used (not between-test errors) because new
    data comes after the training period -> the honest, slightly wider band."""
    return stored_validation()['band']


# ---------------------------------------------------------------------------------------------
# VFM validation tab
# ---------------------------------------------------------------------------------------------
def vfm_validation():
    v = dict(stored_validation())
    tests = data.bundle()['tests']
    v['dataset_tests'] = None
    if tests is not None and len(tests):
        t = tests.copy()
        t['in_training'] = t.time <= pd.Timestamp('2025-09-30 23:59')        # used to fit the stored model?
        v['dataset_tests'] = [{'well': r.well_name, 'time': str(r.time)[:16], 'in_training': bool(r.in_training),
                               'oil': r.oil_rate_stbd, 'water': r.water_rate_stbd, 'gas': r.gas_rate_mscfd,
                               'oil_pred': r.pred_oil_rate_stbd, 'water_pred': r.pred_water_rate_stbd,
                               'gas_pred': r.pred_gas_rate_mscfd} for r in t.itertuples()]
    v['buildup_check'] = buildup_vs_pres()
    for k in ['pta_synthetic']:
        v.pop(k, None)
    return v


def buildup_vs_pres():
    """Independent check of the VFM reservoir pressure: at the end of a long build-up the shut-in pressure
    approaches the reservoir pressure. The VFM infers P_res only from well tests (P_res = Pwf + q / J), so the
    two numbers come from different data. (Without a boundary the pressure is still rising -> lower bound.)"""
    hf, rates = data.high_freq(), data.daily_rates()
    rows = []
    for r in data.pta_results():
        end = r['t_shut'] + pd.Timedelta(hours=r['duration_hr'])
        w = hf[(hf.well_name == r['well_name']) & hf.timestamp.between(end - pd.Timedelta(minutes=30), end)]
        day = rates[(rates.well_name == r['well_name']) & (rates.date == r['t_shut'].normalize() - pd.Timedelta(days=1))]
        if w.empty or day.empty or pd.isna(day.p_res.iloc[0]):
            continue
        p_end, p_res = float(w.dh_pressure_psia.mean()), float(day.p_res.iloc[0])
        rows.append({'well': r['well_name'], 't_shut': r['t_shut'].strftime('%Y-%m-%d'), 'duration_hr': r['duration_hr'],
                     'event_type': r['event_type'], 'boundary': r['model_boundary'], 'p_end_buildup': p_end,
                     'p_res_vfm': p_res, 'difference_psi': p_end - p_res})
    return rows


# ---------------------------------------------------------------------------------------------
# PTA validation tab
# ---------------------------------------------------------------------------------------------
def radial_quality(r):
    """How trustworthy the radial-flow window is.
    - flatness: coefficient of variation of the Bourdet derivative inside the window (std / mean, %);
      radial flow = flat derivative, so a small value means a clean straight line on the Horner plot
    - window width in log cycles: log10(end / start); wider = more points in the Horner fit (our minimum is 0.5)
    - radius of investigation reached during the build-up."""
    dt, der = np.asarray(r['_dt_bins']), np.asarray(r['_derivative'])
    inside = (dt >= r['radial_start_hr']) & (dt <= r['radial_end_hr']) & (der > 0)
    cv = float(np.std(der[inside]) / np.mean(der[inside]) * 100) if inside.sum() >= 3 else None
    return {'flatness_cv_pct': cv, 'log_cycles': float(np.log10(r['radial_end_hr'] / r['radial_start_hr'])),
            'radius_investigation_ft': r['radius_investigation_ft']}


def pta_validation():
    events = []
    checks = {(c['well'], c['t_shut']): c for c in buildup_vs_pres()}
    for r in data.pta_results():
        c = checks.get((r['well_name'], r['t_shut'].strftime('%Y-%m-%d')))
        events.append({'event_id': r['event_id'], 'well': r['well_name'], 't_shut': r['t_shut'].strftime('%Y-%m-%d %H:%M'),
                       'event_type': r['event_type'], 'k_horner': r['k_md'], 'k_model': r['model_k_md'],
                       'skin_horner': r['skin'], 'skin_model': r['model_skin'], 'boundary': r['model_boundary'],
                       'model_rms_pct': r['model_rms_pct'], **radial_quality(r),
                       'p_end_buildup': c['p_end_buildup'] if c else None, 'p_res_vfm': c['p_res_vfm'] if c else None})
    return {'synthetic': stored_validation()['pta_synthetic'], 'events': events}


@lru_cache(maxsize=32)
def _sensitivity(bundle_id, event_id):
    r = data.load_bundle(bundle_id)['pta'][event_id]
    event = {'well_name': r['well_name'], 't_shut': r['t_shut'], 't_open': r['t_open'], 'duration_hr': r['duration_hr']}
    s = pta.sensitivity(data.load_bundle(bundle_id)['hf'], event, r['q_oil_stbd'], r['cum_oil_stb'],
                        data.load_bundle(bundle_id)['pvt'].loc[r['well_name']])
    return s[['case', 'k_change_pct', 'skin_change']].to_dict('records')


def pta_sensitivity(event_id):
    """Each judgement call of the automatic analysis changed on its own (derivative smoothing, minimum window
    width, window start / end, VFM rate q +-5 %): change of k [%] and skin vs the base case."""
    return _sensitivity(data.ACTIVE['id'], event_id)


# ---------------------------------------------------------------------------------------------
# IPR and what-if (VFM page)
# ---------------------------------------------------------------------------------------------
def ipr(well):
    """Inflow performance of one well today, for the what-if card.

    Above the bubble point (straight-line productivity index):   q = J (P_res - Pwf)
    Below the bubble point (Vogel, composite IPR):                q = q_b + (J Pb / 1.8) (1 - 0.2 Pwf/Pb - 0.8 (Pwf/Pb)^2)
                                                                   with q_b = J (P_res - Pb)
    Skin what-if (Darcy):  J ~ 1 / (ln(re/rw) - 0.75 + s)  ->  J_new = J (ln(re/rw) - 0.75 + s) / (ln(re/rw) - 0.75 + s_new)
    Intuition: lowering the flowing pressure (more lift gas, bigger choke) or removing damage moves the operating
    point along / up the inflow curve -> more oil, as long as Pwf stays above the bubble point.
    """
    s = pages.well_summary(well)
    g = data.daily_rates()
    g = g[(g.well_name == well) & (g.on_stream_frac > 0.99) & g.pwf.notna()].sort_values('date').tail(30)
    pvt = data.bundle()['pvt'].loc[well]
    re = np.sqrt(pvt.estimated_drainage_area_acres * 43560 / np.pi)                   # drainage radius [ft]
    ln_term = float(np.log(re / pvt.wellbore_radius_ft) - 0.75)
    J, p_res, pb = s['J'], float(g.p_res.mean()), float(pvt.bubble_point_pressure_psia)
    pwf = np.linspace(0, p_res, 80)
    q_b = J * (p_res - pb)
    q = np.where(pwf >= pb, J * (p_res - pwf), q_b + J * pb / 1.8 * (1 - 0.2 * pwf / pb - 0.8 * (pwf / pb) ** 2))
    return {'well': well, 'J': J, 'p_res': p_res, 'pb': pb, 'pwf_now': float(g.pwf.mean()), 'oil_now': float(g.oil.mean()),
            'water_cut_now': float(g.water_cut.mean()), 'rs': s['rs_scf_stb'], 'skin': s['skin'], 'k_md': s['k_md'],
            'ln_term': ln_term, 'curve': {'pwf': pwf.round(1).tolist(), 'oil': np.clip(q, 0, None).round(1).tolist()}}


# ---------------------------------------------------------------------------------------------
# Production forecast
# ---------------------------------------------------------------------------------------------
def _recent_line(dates, values, days=90):
    """Straight line through the last `days` days: returns (value at the last day, slope per day)."""
    d = pd.DataFrame({'date': pd.to_datetime(dates), 'v': values}).dropna()
    d = d[d.date > d.date.max() - pd.Timedelta(days=days)]
    x = (d.date - d.date.max()).dt.total_seconds() / 86400
    if len(d) < 10:
        return float(d.v.iloc[-1]), 0.0
    slope, intercept = np.polyfit(x, d.v, 1)
    return float(intercept), float(slope)


def forecast(months=6):
    """Rates for the next `months` months if the wells keep today's operating conditions.

    Per well:  P_res(t)  = straight line through the last 90 days of the VFM reservoir pressure (depletion)
               WC(t)     = straight line through the last 90 days of the VFM water cut, kept within 0-99 %
               Pwf       = held at the mean of the last 30 flowing days (same choke and lift gas)
               oil(t)    = J (P_res(t) - Pwf)            water = oil WC / (100 - WC)        gas = oil Rs / 1000
    Uncertainty: the P10 / P90 multipliers of the 0-3-month forecast validation; beyond 3 months the half-width is
    widened in proportion to time (an assumption: errors grow as the trends are extrapolated further).
    """
    rates = data.daily_rates()
    model = data.vfm_model()
    last = rates.date.max()
    horizon = int(months * 30.4)
    future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq='D')
    t = np.arange(1, horizon + 1)
    b = band()
    widen = np.maximum(1.0, t / 92.0)
    wells, field = [], {f: np.zeros(horizon) for f in ['oil', 'water', 'gas']}
    for w in sorted(rates.well_name.unique()):
        if w not in model['trends'].index:
            continue
        g = rates[rates.well_name == w].sort_values('date')
        flow = g[(g.on_stream_frac > 0.99) & g.pwf.notna()]
        J, rs = model['trends'].loc[w, 'J'], float(model['trends'].loc[w, 'gor_coef'][0])
        p0, p_slope = _recent_line(flow.date, flow.p_res)
        wc0, wc_slope = _recent_line(flow.date, flow.water_cut)
        pwf = float(flow.pwf.tail(30).mean())
        p_res = p0 + p_slope * t
        wc = np.clip(wc0 + wc_slope * t, 0, 99)
        oil = np.clip(J * (p_res - pwf), 0, None)
        water, gas = oil * wc / (100 - wc), oil * rs / 1000
        field['oil'] += oil
        field['water'] += water
        field['gas'] += gas
        pb = float(data.bundle()['pvt'].loc[w, 'bubble_point_pressure_psia'])
        wells.append({'well': w, 'oil_now': float(flow.oil.tail(30).mean()), 'oil_3m': float(oil[min(90, horizon - 1)]),
                      'oil_end': float(oil[-1]), 'cum_oil': float(oil.sum()), 'wc_now': wc0, 'wc_end': float(wc[-1]),
                      'p_res_now': p0, 'p_res_end': float(p_res[-1]), 'p_res_slope_month': p_slope * 30.4,
                      'pwf_held': pwf, 'pb': pb, 'pwf_margin_to_pb': pwf - pb})
    hist = rates[rates.date > last - pd.Timedelta(days=120)].groupby('date')[['oil', 'water', 'gas']].sum()
    out = {'months': months, 'start': future[0].strftime('%Y-%m-%d'), 'end': future[-1].strftime('%Y-%m-%d'),
           'history': {'date': hist.index.strftime('%Y-%m-%d').tolist(), **{f: hist[f].round(1).tolist() for f in field}},
           'future': {'date': future.strftime('%Y-%m-%d').tolist()}, 'wells': wells, 'band': b}
    for f, series in field.items():
        lo = 1 - (1 - b[f]['p10']) * widen
        hi = 1 + (b[f]['p90'] - 1) * widen
        out['future'][f] = series.round(1).tolist()
        out['future'][f + '_p10'] = (series * lo).round(1).tolist()
        out['future'][f + '_p90'] = (series * hi).round(1).tolist()
    out['totals'] = {'oil_now': float(sum(w['oil_now'] for w in wells)), 'oil_end': float(field['oil'][-1]),
                     'cum_oil': float(field['oil'].sum()), 'cum_oil_p10': float((field['oil'] * (1 - (1 - b['oil']['p10']) * widen)).sum()),
                     'cum_oil_p90': float((field['oil'] * (1 + (b['oil']['p90'] - 1) * widen)).sum())}
    return out

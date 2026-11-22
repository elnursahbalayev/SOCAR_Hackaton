"""Data layer behind the web API.

Everything the pages show comes from a RESULTS BUNDLE written by src/pipeline.py:
  data/bundles/bundled/   our provided dataset (Jan-Sep 2025)
  data/bundles/<id>/      a dataset uploaded in the app (e.g. the judges' test data)
The ACTIVE bundle can be switched; results are cached per bundle (functools.lru_cache) so pages are instant.
All engineering logic lives in src/ — this file only reads bundles and shapes them for the pages.
"""
import json
import pickle
import sys
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import export          # noqa: E402
import pipeline        # noqa: E402
import pta             # noqa: E402

BUNDLES = ROOT / 'data' / 'bundles'
UPLOADS = ROOT / 'data' / 'uploads'
ACTIVE = {'id': 'bundled'}
JOBS = {}


# ---------------------------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------------------------
def bundle_dir(bundle_id=None):
    return BUNDLES / (bundle_id or ACTIVE['id'])


@lru_cache(maxsize=4)
def load_bundle(bundle_id):
    d = bundle_dir(bundle_id)
    rates = pd.read_parquet(d / 'rates.parquet')
    rates = rates.rename(columns={'dh_pressure_psia': 'pwf', 'choke_opening_pct': 'choke',
                                  'gas_lift_rate_mscfd': 'gas_lift'})
    liquid = rates.oil + rates.water
    rates['water_cut'] = np.where(liquid > 0, 100 * rates.water / liquid, np.nan)
    with open(d / 'pta_results.pkl', 'rb') as fh:
        results = pickle.load(fh)
    for i, r in enumerate(results):
        r['event_id'] = i
    tests = pd.read_parquet(d / 'tests.parquet') if (d / 'tests.parquet').exists() else None
    return {'hf': pd.read_parquet(d / 'high_freq_clean.parquet'), 'rates': rates, 'pta': results,
            'summary': json.loads((d / 'summary.json').read_text()), 'tests': tests,
            'pvt': pd.read_csv(d / 'pvt.csv').set_index('well_name'),
            'completion': pd.read_csv(d / 'completion.csv').set_index('well_name')}


def bundle():
    return load_bundle(ACTIVE['id'])


def high_freq():
    return bundle()['hf']


def daily_rates():
    return bundle()['rates']


def pta_results():
    return bundle()['pta']


@lru_cache
def vfm_model():
    with open(ROOT / 'models' / 'vfm_model.pkl', 'rb') as fh:
        return pickle.load(fh)


@lru_cache
def vfm_validation():
    """Accuracy of the final model (outputs/final_model_metrics.json, written by notebooks/11):
    leave-one-test-out (between tests) and time-forward (forecasting the next 0-3 months)."""
    m = json.loads((ROOT / 'outputs' / 'final_model_metrics.json').read_text())
    n_experiments = sum(len(pd.read_csv(ROOT / 'outputs' / f)) for f in
                        ['ablation_results.csv', 'time_forward_results.csv', 'full_evaluation_results.csv'])
    out = {'name': m['model'], 'n_tests': m['n_tests'], 'n_experiments': n_experiments}
    for fluid in ['oil', 'water', 'gas']:
        out[f'{fluid}_mape'] = m[f'loto_{fluid}_MAPE']
        out[f'{fluid}_wape'] = m[f'loto_{fluid}_WAPE']
        out[f'forward_{fluid}_mape'] = m[f'forward_{fluid}_MAPE']
        out[f'forward_{fluid}_wape'] = m[f'forward_{fluid}_WAPE']
    return out


def dataset_info():
    s = bundle()['summary']
    return {'id': ACTIVE['id'], 'label': 'Provided dataset' if ACTIVE['id'] == 'bundled' else 'Uploaded dataset',
            **{k: s[k] for k in ['created', 'period', 'wells', 'n_days', 'files', 'warnings', 'clean_report',
                                 'unknown_wells', 'accuracy_on_uploaded_tests', 'n_shut_ins', 'n_buildup_tests']}}


def list_datasets():
    out = []
    for d in sorted(BUNDLES.iterdir()):
        if (d / 'summary.json').exists():
            s = json.loads((d / 'summary.json').read_text())
            out.append({'id': d.name, 'created': s['created'], 'period': s['period'], 'n_wells': len(s['wells']),
                        'active': d.name == ACTIVE['id']})
    return out


def activate(bundle_id):
    if not (bundle_dir(bundle_id) / 'summary.json').exists():
        raise KeyError(bundle_id)
    ACTIVE['id'] = bundle_id


# ---------------------------------------------------------------------------------------------
# Upload jobs: run the pipeline in a background thread, the page polls the progress
# ---------------------------------------------------------------------------------------------
def start_upload_job(saved_paths, bundle_id, on_done=None):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {'id': job_id, 'status': 'running', 'step': 'validate', 'log': [], 'error': None,
                    'bundle_id': bundle_id, 'steps': pipeline.STEPS, 'started': datetime.now().isoformat()}

    def progress(step, message):
        JOBS[job_id]['step'] = step
        JOBS[job_id]['log'].append({'step': step, 'message': message, 'time': datetime.now().strftime('%H:%M:%S')})

    def work():
        try:
            pipeline.run(saved_paths, bundle_dir(bundle_id), progress=progress)
            activate(bundle_id)
            JOBS[job_id]['status'] = 'done'
            if on_done:                          # e.g. pre-build the report so the next page opens instantly
                on_done()
        except Exception as exc:                 # shown on the page, never crashes the server
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = str(exc)

    threading.Thread(target=work, daemon=True).start()
    return JOBS[job_id]


# ---------------------------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------------------------
def _pct_change(new, old):
    return float((new / old - 1) * 100) if old else None


def _window_change(daily, col, last, days=30):
    """% change of the mean daily value: last `days` days vs the `days` before.
    Only when BOTH windows are (almost) complete -- a short uploaded period would otherwise compare
    a full month with a partial one and show a fake jump."""
    recent = daily[daily.date > last - pd.Timedelta(days=days)]
    before = daily[(daily.date <= last - pd.Timedelta(days=days)) & (daily.date > last - pd.Timedelta(days=2 * days))]
    if recent.date.nunique() < days - 2 or before.date.nunique() < days - 2:
        return None
    return _pct_change(recent[col].mean(), before[col].mean())


def overview():
    rates = daily_rates()
    last = rates.date.max()
    today = rates[rates.date == last]
    field = rates.groupby('date')[['oil', 'water', 'gas']].sum().reset_index()
    tests = [r for r in pta_results() if r['event_type'] == 'planned build-up test']
    pta_by_well = {r['well_name']: r for r in tests}
    trends = vfm_model()['trends']

    wells = []
    for well, g in rates.groupby('well_name'):
        g = g.sort_values('date')
        t = g.iloc[-1]
        p = pta_by_well.get(well)
        wells.append({
            'well': well, 'status': 'Flowing' if t.on_stream_frac > 0 else 'Shut-in',
            'oil': t.oil, 'water': t.water, 'gas': t.gas, 'water_cut': t.water_cut,
            'gor': 1000 * t.gas / t.oil if t.oil > 0 else None,
            'oil_change_pct': _window_change(g, 'oil', last),
            'p_res': t.p_res,
            'p_res_decline_psi_month': -float(trends.loc[well, 'pres_coef'][0]) * 30.4 if well in trends.index else None,
            'k_md': p['k_md'] if p else None, 'skin': p['skin'] if p else None,
            'spark': g.oil.round(1).tolist(),
        })

    shut_ins = pta_results()
    kpis = {
        'date': last.strftime('%Y-%m-%d'),
        'oil': today.oil.sum(), 'water': today.water.sum(), 'gas': today.gas.sum(),
        'water_cut': 100 * today.water.sum() / (today.oil.sum() + today.water.sum()),
        'oil_change_pct': _window_change(field, 'oil', last),
        'water_change_pct': _window_change(field, 'water', last),
        'gas_change_pct': _window_change(field, 'gas', last),
        'wells_flowing': int((today.on_stream_frac > 0).sum()), 'wells_total': int(today.well_name.nunique()),
        'shut_in_events': len(shut_ins), 'buildup_tests': len(tests),
        'deferred_oil_stb': sum(r['deferred_oil_stb'] for r in shut_ins),
        'cum_oil_period': rates.oil.sum(),
    }
    attention = []
    for r in tests:
        if r['skin'] > 3:
            attention.append({'level': 'high', 'well': r['well_name'], 'title': 'Stimulation candidate',
                              'text': f"Skin {r['skin']:.1f}: {r['dP_skin_psi']:.0f} psi lost near-wellbore, "
                                      f"flow efficiency {r['flow_efficiency']:.2f}"})
        if r['model_boundary'] == 'constant_pressure':
            attention.append({'level': 'info', 'well': r['well_name'], 'title': 'Pressure support detected',
                              'text': f"Constant-pressure boundary at ~{r['model_boundary_ft']:.0f} ft (aquifer / injector)"})
    for w in wells:
        if w['p_res_decline_psi_month'] and w['p_res_decline_psi_month'] > 15:
            attention.append({'level': 'medium', 'well': w['well'], 'title': 'Fast depletion',
                              'text': f"Reservoir pressure falling {w['p_res_decline_psi_month']:.0f} psi/month"})
    return {
        'kpis': kpis,
        'field_daily': field.assign(date=field.date.dt.strftime('%Y-%m-%d')).round(1).to_dict('records'),
        'wells': wells, 'attention': attention,
        'shut_ins': [{'well': r['well_name'], 't_shut': r['t_shut'].strftime('%Y-%m-%d %H:%M'),
                      'duration_hr': r['duration_hr'], 'deferred_oil_stb': r['deferred_oil_stb'],
                      'event_type': r['event_type']} for r in shut_ins],
        'model': vfm_validation(),
        'period': {'start': rates.date.min().strftime('%Y-%m-%d'), 'end': last.strftime('%Y-%m-%d')},
        'dataset': {'id': ACTIVE['id'], 'label': dataset_info()['label']},
    }


# ---------------------------------------------------------------------------------------------
# PTA page
# ---------------------------------------------------------------------------------------------
PTA_SUMMARY_KEYS = ['event_id', 'well_name', 't_shut', 'duration_hr', 'q_oil_stbd', 'tp_hr', 'pwf_psia',
                    'radial_start_hr', 'radial_end_hr', 'm_psi_cycle', 'p_star_psia', 'p1hr_psia', 'k_md',
                    'skin', 'dP_skin_psi', 'flow_efficiency', 'radius_investigation_ft', 'model_boundary',
                    'model_k_md', 'model_skin', 'model_C_bbl_psi', 'model_boundary_ft', 'model_rms_pct',
                    'interpretation', 'deferred_oil_stb', 'event_type', 'classification_reason']


def pta_events():
    return [{k: r[k] for k in PTA_SUMMARY_KEYS} for r in pta_results()]


def _log_thin(n, keep=400):
    """Indices of ~`keep` points spread evenly on a log scale (enough for a smooth log-axis plot)."""
    if n <= keep:
        return np.arange(n)
    return np.unique(np.round(np.logspace(0, np.log10(n), keep)).astype(int) - 1)


def pta_event_detail(event_id, window=None):
    """Everything the PTA page plots for one shut-in. window=(start_hr, end_hr) overrides the radial window."""
    base = pta_results()[event_id]
    r = base
    if window:
        event = {'well_name': base['well_name'], 't_shut': base['t_shut'], 't_open': base['t_open'],
                 'duration_hr': base['duration_hr']}
        r = pta.analyze_event(high_freq(), event, base['q_oil_stbd'], base['cum_oil_stb'],
                              bundle()['pvt'].loc[base['well_name']], window=window,
                              pwf_hours=pta.FINAL_PTA_OPTIONS['pwf_hours'])
        r['interpretation'] = pta.interpret(r['k_md'], r['skin'], r['dP_skin_psi'], r['flow_efficiency'],
                                            base['model_boundary'], base['model_boundary_ft'],
                                            base['_best']['radius_investigation_ft'])
    well, t0 = base['well_name'], base['t_shut']
    t1 = t0 + pd.Timedelta(hours=base['duration_hr'])

    # (a) history: 1.5 days either side, 5-minute averages
    hf = high_freq()
    g = hf[(hf.well_name == well) & (hf.timestamp >= t0 - pd.Timedelta(hours=36)) & (hf.timestamp <= t1 + pd.Timedelta(hours=36))]
    g = g.set_index('timestamp')[['dh_pressure_psia', 'whp_psia', 'gas_lift_rate_mscfd']].resample('5min').mean().dropna(how='all')
    rates = daily_rates()
    rr = rates[(rates.well_name == well) & (rates.date >= (t0 - pd.Timedelta(days=2)).normalize()) & (rates.date <= t1 + pd.Timedelta(days=2))]

    # (b) Horner and (c) log-log: minute data thinned on a log scale
    dt, pws, tp = r['_dt'], r['_pws'], r['tp_hr']
    idx = _log_thin(len(dt))
    ht_line = np.logspace(0, np.log10(((tp + dt) / dt).max()), 60)

    # (d) model match curves
    model_curves = []
    for f in base['_fits']:
        der = pta.bourdet_derivative(base['_dt_fit'], f['_model_dp'], tp)
        model_curves.append({'boundary': f['boundary'], 'chosen': f is base['_best'], 'rms_pct': f['rms_error_pct'],
                             'k_md': f['k_md'], 'skin': f['skin'], 'C_bbl_psi': f['C_bbl_psi'],
                             'boundary_ft': f['boundary_radius_ft'], 'dp': f['_model_dp'], 'der': der})

    summary = {k: base[k] for k in PTA_SUMMARY_KEYS}
    for k in ['radial_start_hr', 'radial_end_hr', 'm_psi_cycle', 'p_star_psia', 'p1hr_psia', 'k_md', 'skin',
              'dP_skin_psi', 'flow_efficiency', 'radius_investigation_ft', 'interpretation']:
        summary[k] = r[k]
    summary['manual_window'] = bool(window)
    auto = {k: base[k] for k in ['radial_start_hr', 'radial_end_hr', 'm_psi_cycle', 'k_md', 'skin', 'dP_skin_psi']}

    return {
        'summary': summary, 'auto': auto,
        'history': {'time': g.index.strftime('%Y-%m-%d %H:%M').tolist(), 'pressure': g.dh_pressure_psia.round(2).tolist(),
                    'whp': g.whp_psia.round(2).tolist(), 'gas_lift': g.gas_lift_rate_mscfd.round(1).tolist(),
                    'rate_date': rr.date.dt.strftime('%Y-%m-%d').tolist(), 'oil_rate': rr.oil.round(1).tolist(),
                    't_shut': t0.strftime('%Y-%m-%d %H:%M'), 't_open': t1.strftime('%Y-%m-%d %H:%M')},
        'buildup': {'dt': dt[idx].round(5).tolist(), 'pws': pws[idx].round(2).tolist(),
                    'horner_time': ((tp + dt[idx]) / dt[idx]).tolist(),
                    'dp': (pws[idx] - r['pwf_psia']).round(3).tolist()},
        'horner_line': {'horner_time': ht_line.tolist(),
                        'pws': (r['p_star_psia'] - r['m_psi_cycle'] * np.log10(ht_line)).round(2).tolist()},
        'derivative': {'dt': r['_dt_bins'].tolist(), 'dp': r['_dp_bins'].round(3).tolist(),
                       'der': np.where(r['_derivative'] > 0, r['_derivative'], np.nan).round(4).tolist()},
        'model': {'dt': base['_dt_fit'].tolist(), 'curves': model_curves},
        'limits': {'dt_min': float(dt.min()), 'dt_max': float(dt.max())},
    }


def excel_bytes(kind):
    """The VFM or PTA results Excel of the active dataset (written by the pipeline)."""
    return (bundle_dir() / f'{kind}_results.xlsx' if kind == 'PTA' else bundle_dir() / 'VFM_predictions.xlsx').read_bytes()


def csv_bytes(kind):
    """The results of the active dataset as CSV (comma-separated, UTF-8 without BOM, LF line endings).
    'VFM'     : the organisers' VFM template: timestamp, well_name, oil_rate_stbd, water_rate_stbd, gas_rate_mscfd
    'PTA'     : the organisers' PTA template (planned build-up tests): well_name, shut_in_time, permeability_md,
                skin, delta_p_skin_psi, horner_slope
    'PTA_all' : every shut-in with its classification (planned / unplanned), reason and results
    Made from the same Excel files as the Excel downloads, so both always agree."""
    if kind == 'VFM':
        df = export.vfm_submission(pd.read_excel(bundle_dir() / 'VFM_predictions.xlsx', sheet_name='predictions'))
    elif kind == 'PTA':
        df = export.pta_submission(pd.read_excel(bundle_dir() / 'PTA_results.xlsx', sheet_name='pta_results'))
    else:
        df = pd.read_excel(bundle_dir() / 'PTA_results.xlsx', sheet_name='all_shut_ins')
    return df.to_csv(index=False, lineterminator=chr(10)).encode('utf-8')   # LF line endings


# ---------------------------------------------------------------------------------------------
# Data page: measurement explorer
# ---------------------------------------------------------------------------------------------
def signals(well, resample='1h'):
    """One well's sensors (resampled) + choke + shut-ins, for the explorer chart."""
    hf = high_freq()
    g = hf[hf.well_name == well].set_index('timestamp')
    cols = ['whp_psia', 'whp_downstream_psia', 'dh_pressure_psia', 'dh_temperature_degf',
            'annulus_pressure_psia', 'gas_lift_rate_mscfd', 'choke_opening_pct']
    s = g[cols].resample(resample).mean()
    out = {'time': s.index.strftime('%Y-%m-%d %H:%M').tolist()}
    for c in cols:
        out[c] = s[c].round(2).tolist()
    out['shut_ins'] = [{'t_shut': r['t_shut'].strftime('%Y-%m-%d %H:%M'), 'duration_hr': r['duration_hr'],
                        'event_type': r['event_type']} for r in pta_results() if r['well_name'] == well]
    return out

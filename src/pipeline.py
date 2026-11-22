"""End-to-end pipeline: raw CSV files -> validated, cleaned data -> VFM rates -> PTA -> Excel results.

The SAME code runs for the bundled dataset and for files uploaded in the app (e.g. the judges' test data), so
the live demo uses exactly the method that produced the deliverables.

    import pipeline
    summary = pipeline.run(['a.csv', 'b.csv'], out_dir, progress=print)

Robustness for unknown test data:
  - file TYPE is recognised from its columns (any file name works); any subset of files may be uploaded
  - missing static files (PVT, completion) -> the bundled ones; choke events are combined with the bundled
    history so the choke state at the start of a new period is known; no choke file -> shut-ins from gas lift
  - PTA in a later period: the 7-day rate and the cumulative oil also use the bundled-period VFM rates
  - wells the model has never seen -> warning, no prediction (the model is calibrated per well)
  - uploaded well tests (optional) -> accuracy of the stored model on them (no retraining)
"""
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import clean
import export
import features as F
import load
import pta
import pta_model
import validation as V
import vfm

STEPS = ['validate', 'clean', 'features', 'vfm', 'pta', 'export']
TRAINING_END = pd.Timestamp('2025-09-30 23:59')   # last day of the data the stored model was trained on

# columns a file must contain to be recognised as that type
FILE_TYPES = {
    'high_freq': ['timestamp', 'well_name', 'whp_psia', 'whp_downstream_psia', 'dh_pressure_psia',
                  'dh_temperature_degf', 'annulus_pressure_psia', 'gas_lift_rate_mscfd'],
    'choke': ['timestamp', 'well_name', 'choke_opening_pct'],
    'well_tests': ['well_name', 'test_start', 'test_end', 'oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd'],
    'pvt': ['well_name', 'net_pay_thickness_ft', 'porosity_frac', 'total_compressibility_1_psi',
            'wellbore_radius_ft', 'oil_fvf_rb_stb', 'oil_viscosity_cp'],
    'completion': ['well_name', 'gauge_tvd_ft', 'cum_oil_at_2025_01_01_stb', 'solution_gor_at_pb_scf_stb'],
}
DATE_COLUMNS = {'high_freq': ['timestamp'], 'choke': ['timestamp'], 'well_tests': ['test_start', 'test_end']}
BUNDLED_BUNDLE = load.ROOT / 'data' / 'bundles' / 'bundled'


# ---------------------------------------------------------------------------------------------
# 1. Validate
# ---------------------------------------------------------------------------------------------
def detect_file_type(path):
    """Recognise a CSV by its header (all required columns present). Returns the type or None."""
    columns = set(pd.read_csv(path, nrows=0).columns.str.strip())
    for name, required in FILE_TYPES.items():
        if set(required) <= columns:
            return name
    return None


def validate(paths):
    """Check every uploaded file. Returns (report, frames) where frames = {type: DataFrame}."""
    report = {'files': [], 'errors': [], 'warnings': []}
    frames = {}
    for path in map(Path, paths):
        kind = detect_file_type(path)
        entry = {'file': path.name, 'type': kind}
        if kind is None:
            report['warnings'].append(f'{path.name}: not recognised (columns do not match any known file) -> ignored')
            report['files'].append(entry)
            continue
        if kind in frames:
            report['warnings'].append(f'{path.name}: a second {kind} file -> rows appended')
        df = pd.read_csv(path, parse_dates=DATE_COLUMNS.get(kind, []))
        df.columns = df.columns.str.strip()
        frames[kind] = pd.concat([frames[kind], df], ignore_index=True) if kind in frames else df
        entry.update({'rows': len(df), 'wells': sorted(df.well_name.astype(str).unique())})
        if kind in DATE_COLUMNS:
            col = DATE_COLUMNS[kind][0]
            bad = df[col].isna().sum()
            entry.update({'start': str(df[col].min()), 'end': str(df[col].max())})
            if bad:
                report['warnings'].append(f'{path.name}: {bad} rows with unreadable dates -> dropped')
        if kind == 'high_freq':
            sensors = [c for c in clean.SENSORS if c in df]
            entry['null_codes'] = int((df[sensors] < -900).sum().sum())
            entry['duplicate_rows'] = int(df.duplicated(['well_name', 'timestamp']).sum())
        report['files'].append(entry)

    if 'high_freq' not in frames:
        report['errors'].append('No high-frequency sensor file found: it is required for VFM and PTA '
                                '(columns: ' + ', '.join(FILE_TYPES['high_freq']) + ')')
    for kind in ['pvt', 'completion']:
        if kind not in frames:
            report['warnings'].append(f'No {kind} file uploaded -> using the bundled one')
    if 'choke' not in frames:
        report['warnings'].append('No choke file uploaded -> bundled choke history used; shut-ins also detected '
                                  'from gas lift stopping')
    return report, frames


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def shut_ins_from_gas_lift(hf_clean, min_minutes=60):
    """Fallback when there is no choke information: a shut-in = gas lift (almost) stopped for >= 60 min.
    Gas lift is always stopped when a gas-lifted well is closed (seen in every shut-in of this field)."""
    off = hf_clean.gas_lift_rate_mscfd < 20
    block = (off != off.shift()) | (hf_clean.well_name != hf_clean.well_name.shift())
    groups = block.cumsum()
    run_length = off.groupby(groups).transform('sum')
    return off & (run_length >= min_minutes)


def combined_rates(new_rates, bundled_rates):
    """Daily oil rates for PTA: the uploaded period + the bundled period before it (for the 7-day mean rate
    and the cumulative oil before a shut-in). Uploaded days win where both exist."""
    if bundled_rates is None:
        return new_rates
    first = new_rates.groupby('well_name').date.min()
    older = bundled_rates[bundled_rates.date < bundled_rates.well_name.map(first).fillna(pd.Timestamp.max)]
    return pd.concat([older, new_rates], ignore_index=True).sort_values(['well_name', 'date'])


# ---------------------------------------------------------------------------------------------
# 2. Run
# ---------------------------------------------------------------------------------------------
def run(paths, out_dir, progress=print, model_path=load.ROOT / 'models' / 'vfm_model.pkl'):
    """Full pipeline on a list of CSV files. Writes a results bundle to out_dir and returns its summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress('validate', 'Checking the uploaded files')
    report, frames = validate(paths)
    if report['errors']:
        raise ValueError('; '.join(report['errors']))
    warnings = list(report['warnings'])
    pvt = frames.get('pvt', load.load_pvt())
    completion = frames.get('completion', load.load_completion())
    bundled_choke = load.load_choke_events()
    choke = pd.concat([bundled_choke, frames['choke']]) if 'choke' in frames else bundled_choke
    choke = choke.drop_duplicates(['well_name', 'timestamp']).sort_values(['well_name', 'timestamp'])

    progress('clean', 'Cleaning sensor data (null codes, duplicates, spikes) and marking shut-ins')
    # sort by well and time first: the spike filter compares each minute with its NEIGHBOURING rows
    hf = frames['high_freq'].dropna(subset=['timestamp']).sort_values(['well_name', 'timestamp']).reset_index(drop=True)
    hf_clean, clean_report = clean.clean_high_freq(hf, choke)
    if 'choke' not in frames:               # no choke information for this period -> fall back on gas lift
        hf_clean['shut_in'] = hf_clean['shut_in'] | shut_ins_from_gas_lift(hf_clean)
    hf_clean.to_parquet(out_dir / 'high_freq_clean.parquet', index=False)

    progress('features', 'Building daily features (physics features, on-stream fraction)')
    wells = F.build_well_table(pvt, completion, load.load_own_k_skin())
    daily = F.add_physics_features(F.build_daily_table(hf_clean), wells)

    progress('vfm', 'Predicting daily oil, water and gas with the stored VFM model')
    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)
    unknown = sorted(set(daily.well_name) - set(model['wells']))
    if unknown:
        warnings.append(f'Wells not seen in training (no calibration, no prediction): {", ".join(unknown)}')
    known = daily[daily.well_name.isin(model['wells'])].copy()
    rates = vfm.predict_rates(model, known)
    predictions = export.vfm_table(known, rates)
    accuracy = None
    if 'well_tests' in frames:
        tests, _ = clean.clean_well_tests(frames['well_tests'])
        tests = tests[tests.well_name.isin(model['wells'])]
        if len(tests):
            table = F.add_physics_features(F.build_test_table(tests, hf_clean), wells)
            table = table[table.dh_pressure_psia.notna()]
            pred = vfm.predict_rates(model, table)
            # measured vs predicted per test (shown on the VFM page)
            tests_out = table[['well_name', 'time', 'test_hours', 'dh_pressure_psia'] + V.TARGETS].copy()
            for t in V.TARGETS:
                tests_out['pred_' + t] = pred[t].values
            tests_out.to_parquet(out_dir / 'tests.parquet', index=False)
            accuracy = {t.split('_')[0]: {k: round(float(v), 3) for k, v in V.official_metrics(table[t], pred[t]).items()}
                        for t in V.TARGETS}
            accuracy['n_tests'] = int(len(table))
            # tests up to the end of the training data were used to fit the model -> in-sample, not a blind check
            accuracy['n_in_sample'] = int((table.time <= TRAINING_END).sum())

    progress('pta', 'Detecting shut-ins, classifying them and running the build-up analysis')
    new_rates = pd.DataFrame({'date': pd.to_datetime(predictions.date), 'well_name': predictions.well_name,
                              'oil': predictions.oil_rate_stbd})
    bundled_rates = None
    if (BUNDLED_BUNDLE / 'rates.parquet').exists() and out_dir.resolve() != BUNDLED_BUNDLE.resolve():
        b = pd.read_parquet(BUNDLED_BUNDLE / 'rates.parquet')
        bundled_rates = pd.DataFrame({'date': b.date, 'well_name': b.well_name, 'oil': b.oil})
    rates_for_pta = combined_rates(new_rates, bundled_rates)
    events = pta.find_shut_ins(hf_clean)
    events = events[events.well_name.isin(model['wells'])].reset_index(drop=True)
    pta_results = pta.analyze_all(hf_clean, events, rates_for_pta, pvt.set_index('well_name'),
                                  completion.set_index('well_name'), choke) if len(events) else []
    for r in pta_results:
        p = pvt.set_index('well_name').loc[r['well_name']]
        fits, best, dt_fit, dp_fit = pta_model.fit_all_models(r, p)
        r.update({'model_boundary': best['boundary'], 'model_k_md': best['k_md'], 'model_skin': best['skin'],
                  'model_C_bbl_psi': best['C_bbl_psi'], 'model_boundary_ft': best['boundary_radius_ft'],
                  'model_rms_pct': best['rms_error_pct'],
                  'interpretation': pta.interpret(r['k_md'], r['skin'], r['dP_skin_psi'], r['flow_efficiency'],
                                                  best['boundary'], best['boundary_radius_ft'],
                                                  best['radius_investigation_ft']),
                  'deferred_oil_stb': r['q_oil_stbd'] * r['duration_hr'] / 24,
                  '_fits': fits, '_best': best, '_dt_fit': dt_fit, '_dp_fit': dp_fit})

    progress('export', 'Writing Excel results')
    daily_out = known[['date', 'well_name', 'on_stream_frac', 'no_data', 'dh_pressure_psia',
                       'choke_opening_pct', 'gas_lift_rate_mscfd']].copy()
    daily_out[['oil', 'water', 'gas']] = rates[vfm.TARGETS].values
    phys = vfm.physics_predict(known[known.dh_pressure_psia.notna()], model['trends']).reindex(known.index)
    daily_out['p_res'] = phys.P_res.values
    daily_out.to_parquet(out_dir / 'rates.parquet', index=False)
    pvt.to_csv(out_dir / 'pvt.csv', index=False)                   # what this bundle was computed with
    completion.to_csv(out_dir / 'completion.csv', index=False)
    with open(out_dir / 'pta_results.pkl', 'wb') as fh:
        pickle.dump(pta_results, fh)
    export.write_vfm_excel(out_dir / 'VFM_predictions.xlsx', predictions,
                           [('model', model['name']), ('trained until', model['trained_until']),
                            ('units', 'oil, water: STB/d; gas: Mscf/d (formation gas)')])
    planned = [r for r in pta_results if r['event_type'] == 'planned build-up test']
    details = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in pta_results])
    with pd.ExcelWriter(out_dir / 'PTA_results.xlsx', engine='openpyxl') as xl:
        if planned:
            export.pta_table(details[details.event_type == 'planned build-up test']).to_excel(
                xl, sheet_name='pta_results', index=False)
        else:
            pd.DataFrame(columns=list(export.PTA_COLUMNS.values())).to_excel(xl, sheet_name='pta_results', index=False)
        (details if len(details) else pd.DataFrame({'note': ['no shut-ins in this data']})).to_excel(
            xl, sheet_name='all_shut_ins', index=False)

    summary = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'period': {'start': str(hf_clean.timestamp.min()), 'end': str(hf_clean.timestamp.max())},
        'wells': sorted(daily.well_name.unique()), 'n_days': int(daily.date.nunique()),
        'files': report['files'], 'warnings': warnings, 'clean_report': clean_report,
        'unknown_wells': unknown, 'accuracy_on_uploaded_tests': accuracy,
        'n_shut_ins': len(pta_results), 'n_buildup_tests': len(planned),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    progress('done', 'Finished')
    return summary


if __name__ == '__main__':
    # Rebuild the 'bundled' results (what the app shows before any upload) from the provided raw files:
    #   python src/pipeline.py
    raw = load.ROOT / 'data' / 'data'
    names = ['high_freq_data.csv', 'choke_events.csv', 'well_tests.csv', 'reservoir_pvt.csv', 'well_completion.csv']
    run([raw / n for n in names], BUNDLED_BUNDLE)

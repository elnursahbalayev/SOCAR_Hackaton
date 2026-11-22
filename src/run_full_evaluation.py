"""Full re-evaluation of EVERY experiment family with the official metrics (R², MAPE, WAPE per fluid).

For each candidate:
  1. leave-one-test-out (interpolation between tests)              -> loto_* columns
  2. time-forward (train before a cutoff, predict next 92 days)    -> fwd_*  columns
  3. normal-day checks (the test-separator blind spot):
       normal_bias_<fluid>_pct : median % difference vs physics on all normal flowing days

Usage (project root):  python src/run_full_evaluation.py
Writes outputs/full_evaluation_results.csv and outputs/full_evaluation_predictions.parquet.
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import features as F
import load
import synthetic as S
import validation as V
import vfm

RESIDUAL_FEATURES = ['tubing_gradient', 'gas_lift_rate_mscfd', 'choke_flow_term', 'choke_p_ratio',
                     'dh_temperature_degf', 'temp_below_res', 'lift_dp', 'days']


def weights(oil, water, gas):
    return {'oil_rate_stbd': oil, 'water_rate_stbd': water, 'gas_rate_mscfd': gas}


def build_methods(daily, wells):
    """name -> method(train, predict). Built inside each worker (methods are closures)."""
    PF, RAW, SAFE = vfm.PHYSICS_FEATURES, vfm.RAW_FEATURES, vfm.SEPARATOR_SAFE_FEATURES
    P = vfm.make_physics_method
    gp_all = vfm.make_ml_method(PF, vfm.gaussian_process, True, 'onehot')
    gp_safe = vfm.make_ml_method(SAFE, vfm.gaussian_process, True, 'onehot')
    m = {}
    # synthetic pretraining (slowest first)
    # (RandomForest 'mixed' was dropped: ~27 min alone, user did not need it)
    for model, mode in [('ExtraTrees', 'synthetic_only'), ('CatBoost', 'finetune'),
                        ('CatBoost', 'mixed'), ('XGBoost', 'finetune'), ('LightGBM', 'mixed'), ('MLP', 'finetune')]:
        m[f'D {model} {mode}'] = S.make_synthetic_method(model, mode, daily, wells)
    # hybrids
    m['C1 residual Ridge OOF'] = vfm.make_residual_method(RESIDUAL_FEATURES, vfm.ridge, True)
    m['C1 residual LightGBM OOF'] = vfm.make_residual_method(RESIDUAL_FEATURES, vfm.default_lgbm, True)
    m['C2 physics-feature LightGBM OOF'] = vfm.make_physics_feature_method(PF, vfm.default_lgbm, True)
    m['C2 physics-feature Ridge OOF'] = vfm.make_physics_feature_method(PF, vfm.ridge, True)
    m['C2 physics-feature GP OOF'] = vfm.make_physics_feature_method(PF, vfm.gaussian_process, True, 'onehot')
    m['C3 eng-targets LightGBM'] = vfm.make_engineered_target_method(PF, vfm.default_lgbm, False)
    m['C3 eng-targets LightGBM + label feats OOF'] = vfm.make_engineered_target_method(PF, vfm.default_lgbm, True, True)
    m['C3 eng-targets Ridge + label feats OOF'] = vfm.make_engineered_target_method(PF, vfm.ridge, True, True)
    # model zoo on original labels
    for name, make in vfm.MODELS.items():
        for set_name, feats in [('raw', RAW), ('physics', PF)]:
            m[f'A {name} {set_name} feats'] = vfm.make_ml_method(feats, make, True, 'onehot')
    m['A GaussianProcess separator-safe feats'] = gp_safe
    m['A MLP physics feats'] = S.make_synthetic_method('MLP', 'real_only', daily, wells)
    # naive + physics
    m['0 naive last test value'] = vfm.time_interpolation
    m['0 naive well mean'] = vfm.well_mean
    m['B physics'] = vfm.physics
    m['B physics rec120'] = P(recency_tau=120)
    m['B physics rec60'] = P(recency_tau=60)
    m['B physics shrink0.3'] = P(pres_shrink=0.3)
    m['B physics rec120 + WC tubing'] = P(recency_tau=120, wc_correction='tubing')
    m['B physics GOR median'] = P(gor_model='median')
    # blends
    m['E2 physics + GP-all water'] = vfm.make_blend_method(vfm.physics, gp_all, weights(0, .5, 0))
    m['E3 physics + GP-all oil&water'] = vfm.make_blend_method(vfm.physics, gp_all, weights(.5, .5, 0))
    m['F1 rec120 + GP-all oil&water'] = vfm.make_blend_method(P(recency_tau=120), gp_all, weights(.5, .5, 0))
    m['F2 rec120 + GP-safe water (current final)'] = vfm.final_method()
    m['rec120 + GP-safe oil&water'] = vfm.make_blend_method(P(recency_tau=120), gp_safe, weights(.5, .5, 0))
    return m


def load_inputs():
    tests = pd.read_parquet(load.PROCESSED / 'test_features.parquet')
    daily = pd.read_parquet(load.PROCESSED / 'daily_features.parquet')
    wells = F.build_well_table(load.load_pvt(), load.load_completion(), load.load_own_k_skin())
    return tests, daily, wells


def normal_day_inputs(tests, daily):
    """Normal flowing days + the physics prediction on them (reference for the normal-day bias check).
"""
    normal = daily[(daily.on_stream_frac == 1) & daily.dh_pressure_psia.notna()]
    phys = vfm.physics_predict(normal, vfm.fit_well_trends(tests, **vfm.FINAL_PHYSICS_OPTIONS))
    return normal, phys


def run_job(name):
    warnings.filterwarnings('ignore')
    tests, daily, wells = load_inputs()
    method = build_methods(daily, wells)[name]

    loto = V.leave_one_test_out(tests, method)
    row = V.official_row(name, tests, loto, prefix='loto_')

    fwd = V.time_forward(tests, method)
    row.update({k: v for k, v in V.official_row(name, tests.loc[fwd.test_index], fwd, prefix='fwd_').items()
                if k != 'experiment'})

    normal, phys = normal_day_inputs(tests, daily)
    pred_normal = method(tests, normal)
    for t in V.TARGETS:
        row[f'normal_bias_{t.split("_")[0]}_pct'] = (np.median(pred_normal[t] / phys[t]) - 1) * 100

    preds = pd.concat([loto.assign(kind='loto', test_index=loto.index),
                       fwd.assign(kind='fwd').drop(columns=['cutoff'])], ignore_index=True)
    # save THIS experiment immediately -> a stopped / crashed run never loses finished results
    pd.DataFrame([row]).to_csv(part_path(name, 'csv'), index=False)
    preds.assign(experiment=name).to_parquet(part_path(name, 'parquet'), index=False)
    print(f'done {name:45s} fwd WAPE {row["fwd_mean_WAPE"]:.2f}  loto WAPE {row["loto_mean_WAPE"]:.2f}', flush=True)
    return name


PARTS = load.ROOT / 'outputs' / 'full_evaluation_parts'


def part_path(name, ext):
    safe = ''.join(ch if ch.isalnum() else '_' for ch in name)
    return PARTS / f'{safe}.{ext}'


def combine_parts():
    """Merge all finished experiment files into the two summary files."""
    rows = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob('*.csv'))], ignore_index=True)
    rows.to_csv(load.ROOT / 'outputs' / 'full_evaluation_results.csv', index=False)
    preds = pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob('*.parquet'))], ignore_index=True)
    preds.to_parquet(load.ROOT / 'outputs' / 'full_evaluation_predictions.parquet', index=False)
    return rows


if __name__ == '__main__':
    PARTS.mkdir(parents=True, exist_ok=True)
    tests, daily, wells = load_inputs()
    names = [n for n in build_methods(daily, wells) if not part_path(n, 'csv').exists()]   # skip finished ones
    print(f'{len(names)} experiments to run', flush=True)
    with ProcessPoolExecutor(max_workers=14) as pool:
        list(pool.map(run_job, names))
    rows = combine_parts()
    cols = ['experiment', 'fwd_mean_WAPE', 'fwd_mean_MAPE', 'fwd_mean_R2', 'loto_mean_WAPE', 'loto_mean_MAPE',
            'normal_bias_oil_pct', 'normal_bias_water_pct']
    print(rows.sort_values('fwd_mean_WAPE')[cols].round(3).to_string(index=False))

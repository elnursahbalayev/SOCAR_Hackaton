"""Run every approach-D experiment (model x training mode) with leave-one-test-out, in parallel.

Usage (from the project root):
    python src/run_synthetic_experiments.py
Writes:
    outputs/synthetic_results.csv          one row per experiment (MAPE, R2 per target)
    outputs/synthetic_predictions.parquet  LOTO predictions of every experiment (for paired tests)
Takes ~25 minutes on 16 cores (random forest is the slowest).
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')        # one thread per job: the jobs themselves run in parallel
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import pandas as pd

import features as F
import load
import synthetic as S
import validation as V

MODELS = ['Ridge', 'kNN', 'ExtraTrees', 'RandomForest', 'GaussianProcess', 'LightGBM', 'XGBoost', 'CatBoost', 'MLP']


def all_jobs():
    jobs = []
    for model in MODELS:
        modes = ['synthetic_only', 'mixed']
        if model in S.FINETUNABLE:
            modes.append('finetune')
        if model == 'MLP':
            modes.append('real_only')            # the neural net was not in the earlier model comparison
        jobs += [(model, mode) for mode in modes]
    return jobs


def run_job(job):
    warnings.filterwarnings('ignore')
    model, mode = job
    tests = pd.read_parquet(load.PROCESSED / 'test_features.parquet')
    daily = pd.read_parquet(load.PROCESSED / 'daily_features.parquet')
    wells = F.build_well_table(load.load_pvt(), load.load_completion(), load.load_own_k_skin())
    name = f'D {model} {mode}'
    results = []
    predictions, _ = V.run_experiment(name, tests, S.make_synthetic_method(model, mode, daily, wells), results)
    print(f'done: {name:35s} mean MAPE {results[0]["mean_MAPE"]:.2f}', flush=True)
    return results[0], predictions.assign(experiment=name)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=14) as pool:
        outputs = list(pool.map(run_job, all_jobs()))
    rows = pd.DataFrame([o[0] for o in outputs])
    preds = pd.concat([o[1] for o in outputs]).rename_axis('test_index').reset_index()
    rows.to_csv(load.ROOT / 'outputs' / 'synthetic_results.csv', index=False)
    preds.to_parquet(load.ROOT / 'outputs' / 'synthetic_predictions.parquet', index=False)
    print(rows.sort_values('mean_MAPE').round(2).to_string(index=False))

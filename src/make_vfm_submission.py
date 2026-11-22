"""Build a versioned VFM benchmark submission from the final model with chosen physics options.

    import make_vfm_submission as M
    M.make(version=2, physics_options={'recency_tau': 120, 'gor_model': 'rs'})

Writes outputs/submission/vfm_submission_v<version>.csv (exact template format).
Remember to log every version in outputs/submission/SUBMISSION_LOG.md.
"""
import pandas as pd

import export
import load
import vfm


def make(version, physics_options=None, name=None):
    tests = pd.read_parquet(load.PROCESSED / 'test_features.parquet')
    daily = pd.read_parquet(load.PROCESSED / 'daily_features.parquet')
    model = vfm.train_final_model(tests, physics_options, name or f'v{version}')
    rates = vfm.predict_rates(model, daily)
    table = export.vfm_table(daily, rates)
    submission = export.vfm_submission(table)
    path = load.ROOT / 'outputs' / 'submission' / f'vfm_submission_v{version}.csv'
    export.write_submission(submission, path)
    return submission, path

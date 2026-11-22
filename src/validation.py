"""Validation: leave-one-test-out (LOTO) + error metrics.

Why LOTO:
  The Excel deliverable asks for daily rates over the SAME period as the well tests (Jan-Sep 2025).
  So the real task is "estimate the rate on a day between tests". LOTO mimics exactly that:
  hide one test, train on the other 84, predict the hidden one. Repeat for all 85 tests.
  Every test is predicted by a model that never saw it -> honest error estimate.

A 'method' is any function  method(train_table, predict_table) -> DataFrame with TARGETS columns.
Anything fitted from labels (trends, models) must happen INSIDE the method, using train_table only.
"""
import numpy as np
import pandas as pd

TARGETS = ['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd']


def leave_one_test_out(table, method):
    """Predict every test with a model trained on all the other tests. Returns predictions (same index)."""
    predictions = []
    for i in table.index:
        train = table.drop(index=i)
        hidden = table.loc[[i]]
        predictions.append(method(train, hidden))
    return pd.concat(predictions)


def official_metrics(true, pred):
    """The three metrics the organisers score (per fluid): R² /3, MAPE /2, WAPE /5 points.

    R²   = 1 − Σ(pred − true)² / Σ(true − mean)²     share of variation explained (1 = perfect)
    MAPE = mean(|pred − true| / true) · 100           average % error, every row counts equally
    WAPE = Σ|pred − true| / Σ|true| · 100              total error / total volume -> big wells dominate
    """
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    err = pred - true
    return {'R2': 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum(),
            'MAPE': np.mean(np.abs(err) / true) * 100,
            'WAPE': np.abs(err).sum() / np.abs(true).sum() * 100}


def official_row(name, true_table, predictions, prefix=''):
    """One results row: R2 / MAPE / WAPE per fluid + their means across the three fluids."""
    row = {'experiment': name}
    for metric in ['R2', 'MAPE', 'WAPE']:
        row[f'{prefix}mean_{metric}'] = 0.0
    for t in TARGETS:
        m = official_metrics(true_table[t].values, predictions[t].values)
        short = t.split('_')[0]
        for metric, value in m.items():
            row[f'{prefix}{short}_{metric}'] = value
            row[f'{prefix}mean_{metric}'] += value / len(TARGETS)
    return row


def time_forward(table, method, cutoffs=('2025-06-01', '2025-07-01'), horizon_days=92):
    """Forecast-style validation: train only on tests BEFORE a cutoff date, predict the tests after it.

    This mimics the real use: the model is built on past data and must predict future months
    (e.g. trained on Jan-Sep, used on Oct-Dec). Two cutoffs are used so the result does not depend
    on one particular split. Returns predictions with extra columns: cutoff, horizon_days
    (how far after the cutoff the test is), and the true index of the test in `table`.
    """
    out = []
    for cutoff in pd.to_datetime(list(cutoffs)):
        train = table[table.time < cutoff]
        future = table[(table.time >= cutoff) & (table.time < cutoff + pd.Timedelta(days=horizon_days))]
        pred = method(train, future)
        pred['cutoff'] = cutoff
        pred['horizon_days'] = (future.time - cutoff).dt.days.values
        pred['test_index'] = future.index
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def score_time_forward(table, predictions):
    """MAPE per target over all (cutoff, test) pairs, plus per horizon bucket (0-30, 30-60, 60-92 days)."""
    true = table.loc[predictions.test_index, TARGETS].reset_index(drop=True)
    ape = (predictions[TARGETS].reset_index(drop=True) - true).abs() / true * 100
    bucket = pd.cut(predictions.horizon_days.values, [-1, 30, 60, 92], labels=['0-30 d', '30-60 d', '60-92 d'])
    overall = ape.mean().rename('all')
    by_horizon = ape.groupby(bucket, observed=True).mean().T
    return pd.concat([overall, by_horizon], axis=1)


def run_time_forward(name, table, method, results=None):
    """Time-forward validation + one summary row for the results table."""
    predictions = time_forward(table, method)
    s = score_time_forward(table, predictions)
    row = {'experiment': name}
    for t in TARGETS:
        short = t.split('_')[0]
        row[f'{short}_MAPE'] = s.loc[t, 'all']
    row['mean_MAPE'] = s['all'].mean()
    for h in s.columns[1:]:
        row[f'mean_MAPE_{h}'] = s[h].mean()
    if results is not None:
        results.append(row)
    return predictions, s


def score(table, predictions):
    """Error metrics per target.

    MAPE = mean(|pred - true| / true) * 100      average % error (easy to explain, scale-free)
    MAE  = mean(|pred - true|)                   average error in rate units (STB/d or Mscf/d)
    RMSE = sqrt(mean((pred - true)^2))           like MAE but punishes big misses more
    R2   = 1 - SSE / SST                         share of the variation explained (1 = perfect)
    """
    rows = {}
    for t in TARGETS:
        true, pred = table[t], predictions[t]
        err = pred - true
        rows[t] = {
            'MAPE_%': (err.abs() / true).mean() * 100,
            'MAE': err.abs().mean(),
            'RMSE': np.sqrt((err ** 2).mean()),
            'R2': 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum(),
        }
    return pd.DataFrame(rows).T


def score_per_well(table, predictions):
    """MAPE (%) per well and target -> shows which wells are hard."""
    ape = (predictions[TARGETS] - table[TARGETS]).abs() / table[TARGETS] * 100
    return ape.groupby(table['well_name']).mean()


def paired_test(table, predictions_a, predictions_b):
    """Is method B really better than method A, or just lucky? (same 85 tests, compared one by one)

    For each test: d = |% error of B| - |% error of A|   (negative = B better on that test)
    Wilcoxon signed-rank test: are the d's centred below 0 more than chance would allow?
    p < 0.05 -> the improvement is probably real. p > 0.05 -> could easily be luck.
    """
    from scipy.stats import wilcoxon
    rows = {}
    for t in TARGETS:
        d = ((predictions_b[t] - table[t]).abs() - (predictions_a[t] - table[t]).abs()) / table[t] * 100
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows[t] = {'mean_diff_pts': d.mean(), 'B_better_on': int((d < 0).sum()), 'n': len(d), 'p_value': p}
    return pd.DataFrame(rows).T


def save_results(results, path):
    """Add experiment rows to the ablation CSV. Re-running an experiment replaces its old row."""
    new = pd.DataFrame(results)
    try:
        old = pd.read_csv(path)
        old = old[~old.experiment.isin(new.experiment)]
        new = pd.concat([old, new], ignore_index=True)
    except FileNotFoundError:
        pass
    new.to_csv(path, index=False)
    return new


def run_experiment(name, table, method, results=None):
    """LOTO + score, and append one summary row to `results` (list of dicts) for the ablation table."""
    predictions = leave_one_test_out(table, method)
    s = score(table, predictions)
    row = {'experiment': name}
    for t in TARGETS:
        short = t.split('_')[0]                       # oil / water / gas
        row[f'{short}_MAPE'] = s.loc[t, 'MAPE_%']
        row[f'{short}_R2'] = s.loc[t, 'R2']
    row['mean_MAPE'] = s['MAPE_%'].mean()
    if results is not None:
        results.append(row)
    return predictions, s

"""Clean the per-minute sensor data.

Problems found in notebooks/01_eda.ipynb and how we fix them:
  1. Null code -999.25 (standard oilfield "no reading" value)  -> set to NaN
  2. Exact duplicate rows (same well + minute recorded twice)  -> drop
  3. Spikes lasting 1-3 minutes (e.g. downhole pressure 9000 psia) -> set to NaN
     using a rolling-median filter (explained in remove_spikes)
  4. Add choke opening to every minute and flag shut-ins (choke = 0 %)

"Frozen" sensors were checked: the only long flat runs are gas lift = 0 during shut-ins,
which is real, so nothing is removed for that.
"""
import pandas as pd

SENSORS = ['whp_psia', 'whp_downstream_psia', 'dh_pressure_psia',
           'dh_temperature_degf', 'annulus_pressure_psia', 'gas_lift_rate_mscfd']


def remove_null_codes(df):
    """-999.25 is the LAS-format null value. Anything below -900 is treated as 'no reading'."""
    df = df.copy()
    df[SENSORS] = df[SENSORS].mask(df[SENSORS] < -900)
    return df


def drop_duplicate_rows(df):
    """Same well, same minute, identical values -> keep one."""
    return df.drop_duplicates(subset=['well_name', 'timestamp']).reset_index(drop=True)


def remove_spikes(df, window=11, n_noise=20):
    """Remove short sensor spikes with a rolling-median filter.

    For each sensor and well:
        median_t = median of the `window` minutes centred on t   (11 min -> 5 before, 5 after)
        dev_t    = |value_t - median_t|
        noise    = typical dev (median of dev over the whole well) ~ normal sensor jitter
        spike if dev_t > n_noise * noise   -> set to NaN

    Why a MEDIAN (not a mean):
      - A 1-3 minute spike is a minority of the 11 points, so the median ignores it.
      - A real step change (choke closes, pressure builds up) is preserved: once more than half
        of the window is on the new level, the median follows it. A smooth monotonic rise
        (build-up curve) has median = the point itself, so dev ~ 0 and nothing is removed.
    """
    df = df.copy()
    for col in SENSORS:
        median = df.groupby('well_name')[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=3).median())
        dev = (df[col] - median).abs()
        noise = dev.groupby(df['well_name']).transform('median')
        is_spike = dev > n_noise * noise
        df.loc[is_spike, col] = None
    return df


def add_choke_and_shut_in(df, choke):
    """Attach the choke opening (%) valid at each minute and flag shut-ins.

    choke_events.csv only lists CHANGES, so each value holds until the next change
    ('as-of' join = take the latest event at or before this minute).
    shut_in = True when choke opening is 0 % (well closed, no flow).
    """
    df = df.sort_values('timestamp')
    df = pd.merge_asof(df, choke.sort_values('timestamp'), on='timestamp', by='well_name')
    df['shut_in'] = df['choke_opening_pct'] == 0
    return df.sort_values(['well_name', 'timestamp']).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Well tests (labels)
# ---------------------------------------------------------------------------------------------
M3_PER_MSCF = 28.3168   # 1 Mscf = 1000 ft3 = 28.3168 m3

RATE_COLUMNS = ['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd', 'total_gas_rate_mscfd',
                'liquid_rate_stbd', 'gas_lift_rate_mscfd']


def check_well_tests(t):
    """Consistency residuals between measured rates and derived columns (all should be ~0).

    liquid   = oil + water                          [STB/d]
    WC       = 100 * water / (oil + water)          [%]
    GOR      = 1000 * gas / oil                     [scf/STB]   (gas in Mscf/d -> x1000 = scf/d)
    total    = formation gas + lift gas             [Mscf/d]
    GLR      = 1000 * total_gas / liquid            [scf/STB]
    """
    r = pd.DataFrame(index=t.index)
    r['liquid'] = t.liquid_rate_stbd - (t.oil_rate_stbd + t.water_rate_stbd)
    r['water_cut'] = t.water_cut_pct - 100 * t.water_rate_stbd / (t.oil_rate_stbd + t.water_rate_stbd)
    r['gor'] = t.gor_scf_stb - 1000 * t.gas_rate_mscfd / t.oil_rate_stbd
    r['total_gas'] = t.total_gas_rate_mscfd - (t.gas_rate_mscfd + t.gas_lift_rate_mscfd)
    r['glr'] = t.glr_scf_stb - 1000 * t.total_gas_rate_mscfd / t.liquid_rate_stbd
    return r


def clean_well_tests(tests):
    """Fix the errors found in the well-test labels. Returns (clean_tests, report).

    Fix 1 - sign flips: a rate can never be negative. In every negative case the other columns
            prove the true value is the positive one (e.g. liquid = oil + |water|). -> abs()
    Fix 2 - gas in wrong unit: some rows store gas in Sm3/d instead of Mscf/d. Detected when
            gas / (oil * GOR / 1000) ~ 28.3 (= m3 per Mscf). -> divide by 28.3168.
            Same check for total gas against (gas + lift gas).
    """
    t = tests.copy()
    report = {}

    report['sign_flips'] = []
    for col in RATE_COLUMNS:
        for i in t.index[t[col] < 0]:
            report['sign_flips'].append((t.well_name[i], str(t.test_start[i]), col))
    t[RATE_COLUMNS] = t[RATE_COLUMNS].abs()

    gas_from_gor = t.oil_rate_stbd * t.gor_scf_stb / 1000
    gas_in_m3 = (t.gas_rate_mscfd / gas_from_gor).between(25, 32)
    t.loc[gas_in_m3, 'gas_rate_mscfd'] /= M3_PER_MSCF

    total_in_m3 = (t.total_gas_rate_mscfd / (t.gas_rate_mscfd + t.gas_lift_rate_mscfd)).between(25, 32)
    t.loc[total_in_m3, 'total_gas_rate_mscfd'] /= M3_PER_MSCF

    report['gas_unit_fixed'] = t.loc[gas_in_m3, ['well_name', 'test_start']].astype(str).values.tolist()
    report['total_gas_unit_fixed'] = int(total_in_m3.sum())
    report['max_abs_residual_after'] = check_well_tests(t).abs().max().round(2).to_dict()
    return t, report


def clean_high_freq(hf, choke):
    """Run all cleaning steps in order. Returns (clean_df, report) where report counts what was changed."""
    report = {}
    report['null_codes'] = (hf[SENSORS] < -900).sum().to_dict()
    df = remove_null_codes(hf)

    n_before = len(df)
    df = drop_duplicate_rows(df)
    report['duplicates_dropped'] = n_before - len(df)

    nan_before = df[SENSORS].isna().sum()
    df = remove_spikes(df)
    report['spikes_removed'] = (df[SENSORS].isna().sum() - nan_before).to_dict()

    df = add_choke_and_shut_in(df, choke)
    report['shut_in_minutes'] = int(df['shut_in'].sum())
    return df, report

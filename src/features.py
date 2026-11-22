"""Feature engineering for VFM.

Two tables are built with the SAME summary function, so training and prediction see identical features:
  - test table  : one row per well test  (features over [test_start, test_end) + measured rates = labels)
  - daily table : one row per well per calendar day (what we must predict for the Excel deliverable)

Feature groups
  1. Sensor averages over flowing minutes (pressures, temperature, gas lift, choke)
  2. Physics features (choke flow, tubing gradient, Darcy productivity index ...)  -> add_physics_features
  3. Time (reservoir depletes -> rates decline)
  4. Static well properties (PVT / completion)

Label-based features (reservoir-pressure trend, water-cut / GOR from neighbouring tests) are NOT made here:
they use the labels, so they must be fitted inside cross-validation to avoid leaking the answer.
"""
import numpy as np
import pandas as pd

SENSORS = ['whp_psia', 'whp_downstream_psia', 'dh_pressure_psia',
           'dh_temperature_degf', 'annulus_pressure_psia', 'gas_lift_rate_mscfd']
START = pd.Timestamp('2025-01-01')
MINUTES_PER_DAY = 1440


# ---------------------------------------------------------------------------------------------
# 1. Summarise per-minute data over a time window
# ---------------------------------------------------------------------------------------------
def summarize_minutes(minutes):
    """Summary of the per-minute rows of ONE well over ONE window (a test or a day).

    Means are taken over FLOWING minutes only (shut-in minutes would mix 'no flow' pressures in).
    std of downhole pressure = how steady the well was (unstable flow -> noisier pressure).
    """
    flowing = minutes[~minutes['shut_in']]
    out = flowing[SENSORS].mean()
    out['choke_opening_pct'] = flowing['choke_opening_pct'].mean()
    out['dh_pressure_std'] = flowing['dh_pressure_psia'].std()
    out['shut_in_minutes'] = int(minutes['shut_in'].sum())
    return out


def build_test_table(tests, hf):
    """One row per well test: features over the test window + the measured rates (labels)."""
    rows = []
    for _, test in tests.iterrows():
        window = hf[(hf.well_name == test.well_name) &
                    (hf.timestamp >= test.test_start) & (hf.timestamp < test.test_end)]
        row = summarize_minutes(window)
        row['well_name'] = test.well_name
        row['time'] = test.test_start + (test.test_end - test.test_start) / 2   # middle of the test
        row['on_stream_frac'] = 1.0                                            # tests never include shut-ins
        row['test_hours'] = (test.test_end - test.test_start).total_seconds() / 3600
        for label in ['oil_rate_stbd', 'water_rate_stbd', 'gas_rate_mscfd',
                      'liquid_rate_stbd', 'water_cut_pct', 'gor_scf_stb', 'bsw_pct']:
            row[label] = test[label]
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_table(hf):
    """One row per well per calendar day.

    on_stream_frac = fraction of the day the well was flowing = 1 - shut-in minutes / 1440.
    Daily rate = (rate while flowing) x on_stream_frac, e.g. shut in for 6 h -> 0.75 of the flowing rate.
    A fully shut-in day has no flowing minutes -> NaN features -> predicted rate 0.
    """
    hf = hf.assign(date=hf.timestamp.dt.floor('D'))
    rows = []
    for (well, date), minutes in hf.groupby(['well_name', 'date']):
        row = summarize_minutes(minutes)
        row['well_name'] = well
        row['date'] = date
        row['time'] = date + pd.Timedelta(hours=12)                            # middle of the day
        row['on_stream_frac'] = 1 - row['shut_in_minutes'] / MINUTES_PER_DAY
        rows.append(row)
    return fill_missing_days(pd.DataFrame(rows))


def fill_missing_days(daily):
    """Days with NO sensor data at all (communication outage, e.g. GL-06 on 2025-06-24) still need a
    prediction in the deliverable. Sensors change slowly, so we fill them by linear interpolation in time
    from the neighbouring days of the same well, and mark them with no_data = True.
    (Fully shut-in days are different: they have data, the well is just closed -> they stay NaN, rate 0.)
    """
    all_days = pd.date_range(daily.date.min(), daily.date.max(), freq='D')
    full = pd.MultiIndex.from_product([sorted(daily.well_name.unique()), all_days], names=['well_name', 'date'])
    daily = daily.set_index(['well_name', 'date']).reindex(full).reset_index()

    no_data = daily['time'].isna()
    daily['no_data'] = no_data
    daily['time'] = daily['date'] + pd.Timedelta(hours=12)
    daily.loc[no_data, 'shut_in_minutes'] = 0
    daily.loc[no_data, 'on_stream_frac'] = 1.0

    cols = SENSORS + ['choke_opening_pct', 'dh_pressure_std']
    interpolated = daily.groupby('well_name')[cols].transform(lambda s: s.interpolate(limit_direction='both'))
    daily.loc[no_data, cols] = interpolated.loc[no_data]
    return daily


# ---------------------------------------------------------------------------------------------
# 2. Static well properties + Darcy productivity index
# ---------------------------------------------------------------------------------------------
def build_well_table(pvt, completion, k_skin=None):
    """Per-well constants, incl. the Darcy productivity index J.

    Pseudo-steady-state radial inflow (field units):
        J = k * h / (141.2 * Bo * mu * (ln(re / rw) - 0.75 + s))        [STB/d per psi]
        re = sqrt(A * 43560 / pi)     drainage radius [ft] from area A [acres] (1 acre = 43,560 ft2)
    Intuition: thick (h), permeable (k) rock with thin oil (low mu) and no damage (low s) -> high J.

    k_skin: OUR OWN build-up results (DataFrame indexed by well_name with k_md, skin; see
            own_well_properties.py). Wells without a shut-in in the data have no k / skin of their own:
            they get the field-typical value (median k and skin of the analysed wells) as a PRIOR, and
            J_from_pta = False tells the VFM to fit J from that well's tests instead (j_source='auto').
            k_skin=None -> no PTA yet (first step of the bootstrap): J_darcy is NaN for every well.
    """
    w = pvt.set_index('well_name').join(completion.set_index('well_name'))
    w['k_md'], w['skin'], w['J_from_pta'] = np.nan, np.nan, False
    if k_skin is not None and len(k_skin):
        w.loc[k_skin.index, ['k_md', 'skin']] = k_skin[['k_md', 'skin']].values
        w['J_from_pta'] = w.k_md.notna()
        w['k_md'] = w.k_md.fillna(k_skin.k_md.median())
        w['skin'] = w.skin.fillna(k_skin.skin.median())
    w['drainage_radius_ft'] = np.sqrt(w.estimated_drainage_area_acres * 43560 / np.pi)
    w['J_darcy'] = (w.k_md * w.net_pay_thickness_ft /
                    (141.2 * w.oil_fvf_rb_stb * w.oil_viscosity_cp *
                     (np.log(w.drainage_radius_ft / w.wellbore_radius_ft) - 0.75 + w.skin)))
    keep = ['J_darcy', 'J_from_pta', 'k_md', 'skin', 'net_pay_thickness_ft', 'oil_viscosity_cp', 'oil_fvf_rb_stb',
            'bubble_point_pressure_psia', 'reservoir_temperature_degf', 'gauge_tvd_ft', 'gl_valve_tvd_ft',
            'tubing_id_in', 'choke_max_size_64ths', 'solution_gor_at_pb_scf_stb']
    return w[keep].reset_index()


# ---------------------------------------------------------------------------------------------
# 3. Physics features
# ---------------------------------------------------------------------------------------------
def add_physics_features(df, wells):
    """Add physics-based features. Each one is a quantity that physically controls the flow rate.

    choke_dp           = WHP - P_downstream                     pressure drop that pushes fluid through the choke
    choke_p_ratio      = P_downstream / WHP                     < ~0.55 -> critical flow (rate independent of downstream P)
    choke_size_64      = opening% / 100 * max size              bean size in 1/64 inch
    choke_flow_term    = size^2 * sqrt(choke_dp)                orifice equation: q ~ area * sqrt(dP)
    gilbert_term       = WHP * size^1.89                        Gilbert critical choke flow: q ~ WHP * S^1.89 / GLR^0.546
    tubing_dp          = Pwf - WHP                              mostly weight of the fluid column (+ friction)
    tubing_gradient    = tubing_dp / gauge depth  [psi/ft]      mixture density: water 0.433, oil ~0.37, gas-lifted lower
    lift_dp            = P_annulus - WHP                        how hard lift gas is pushed into the tubing
    temp_below_res     = T_res - T_downhole                     flowing fluid keeps the gauge hot: higher rate -> smaller gap
    J_x_pwf            = J * Pwf                                with q = J * (P_res - Pwf): the Pwf part of the inflow equation
    days               = days since 2025-01-01                  reservoir pressure falls with time (depletion)
    """
    df = df.merge(wells, on='well_name', how='left')
    df['choke_dp'] = df.whp_psia - df.whp_downstream_psia
    df['choke_p_ratio'] = df.whp_downstream_psia / df.whp_psia
    df['choke_size_64'] = df.choke_opening_pct / 100 * df.choke_max_size_64ths
    df['choke_flow_term'] = df.choke_size_64 ** 2 * np.sqrt(df.choke_dp.clip(lower=0))
    df['gilbert_term'] = df.whp_psia * df.choke_size_64 ** 1.89
    df['tubing_dp'] = df.dh_pressure_psia - df.whp_psia
    df['tubing_gradient'] = df.tubing_dp / df.gauge_tvd_ft
    df['lift_dp'] = df.annulus_pressure_psia - df.whp_psia
    df['temp_below_res'] = df.reservoir_temperature_degf - df.dh_temperature_degf
    df['J_x_pwf'] = df.J_darcy * df.dh_pressure_psia
    df['days'] = (df.time - START).dt.total_seconds() / 86400
    return df

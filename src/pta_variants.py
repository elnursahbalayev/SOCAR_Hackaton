"""PTA setup variants for benchmark tests. Each variant changes ONE analysis choice from the v5 Horner method.

  A  Pwf(dt=0) = mean of the last 24 h before shut-in        (instead of the last flowing minute)
  B  q = mean VFM oil rate of the last 7 days before shut-in  (instead of the day before)
  C  tp = hours since the start of the data (2025-01-01)      (instead of 24 * cumulative oil / q)
  D  radial window = dt from 1 h to the end of the build-up  (instead of the flattest derivative)
  E  P1hr = MEASURED pressure at dt = 1 h                     (instead of the Horner line at 1 h)
  F  pressures averaged per hour before the analysis          (instead of per-minute data)
Extra options (combine freely): pwf_hours (mean Pwf of the last N hours), pwf_prev_day (mean Pwf of the
previous calendar day), q_days (mean VFM oil rate of the last N days).
"""
import numpy as np
import pandas as pd

import load
import pta

START = pd.Timestamp('2025-01-01')


def analyze(hf, event, vfm_rates, pvt, completion, variant='', pwf_hours=None, pwf_prev_day=False, q_days=None):
    """Horner analysis of one shut-in with one setup choice changed. vfm_rates: date, well_name, oil."""
    p = pvt.loc[event['well_name']]
    r = vfm_rates[vfm_rates.well_name == event['well_name']]
    day0 = event['t_shut'].normalize()
    q = r[r.date == day0 - pd.Timedelta(days=1)].oil.iloc[0]
    if variant == 'B':
        q_days = 7
    if q_days:                                            # average daily rate of the last q_days days
        q = r[(r.date < day0) & (r.date >= day0 - pd.Timedelta(days=q_days))].oil.mean()
    cum = completion.loc[event['well_name'], 'cum_oil_at_2025_01_01_stb'] + r[r.date < day0].oil.sum()

    dt, pws, pwf = pta.buildup_data(hf, event)
    if variant == 'A':
        pwf_hours = 24
    well_hf = hf[hf.well_name == event['well_name']]
    if pwf_hours:                                         # mean flowing pressure of the last pwf_hours hours
        w = well_hf[(well_hf.timestamp < event['t_shut']) &
                    (well_hf.timestamp >= event['t_shut'] - pd.Timedelta(hours=pwf_hours))]
        pwf = w.dh_pressure_psia.mean()
    if pwf_prev_day:                                      # mean pressure of the previous calendar day
        w = well_hf[(well_hf.timestamp >= day0 - pd.Timedelta(days=1)) & (well_hf.timestamp < day0)]
        pwf = w.dh_pressure_psia.mean()
    if variant == 'F':                                    # hourly averages
        hours = np.floor(dt).astype(int)
        frame = pd.DataFrame({'h': hours, 'dt': dt, 'p': pws}).groupby('h').mean()
        dt, pws = frame.dt.values, frame.p.values

    tp = pta.producing_time(cum, q)
    if variant == 'C':
        tp = (event['t_shut'] - START).total_seconds() / 3600

    dt_b, p_b = pta.log_bins(dt, pws)
    der = pta.bourdet_derivative(dt_b, p_b - pwf, tp)
    window = (1.0, dt.max()) if variant == 'D' else pta.find_radial_window(dt_b, der)
    m, p_star, p1hr = pta.fit_horner(dt, pws, tp, window)
    if variant == 'E':
        p1hr = np.interp(1.0, dt, pws)
    k, skin, dp_skin = pta.reservoir_properties(m, p1hr, pwf, q, p)
    return {'well_name': event['well_name'], 'shut_in_time': event['t_shut'].strftime('%Y-%m-%d %H:%M:%S'),
            'permeability_md': round(k, 2), 'skin': round(skin, 2), 'delta_p_skin_psi': round(dp_skin, 2),
            'horner_slope': round(m, 2)}


def build(vfm_submission_path, variant='', **options):
    hf = load.load_high_freq_clean()
    pvt = load.load_pvt().set_index('well_name')
    completion = load.load_completion().set_index('well_name')
    v = pd.read_csv(vfm_submission_path)
    rates = pd.DataFrame({'date': pd.to_datetime(v.timestamp), 'well_name': v.well_name, 'oil': v.oil_rate_stbd})
    rows = [analyze(hf, e, rates, pvt, completion, variant, **options) for _, e in pta.find_shut_ins(hf).iterrows()]
    return pd.DataFrame(rows).sort_values(['well_name', 'shut_in_time']).reset_index(drop=True)

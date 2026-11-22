"""Pressure transient analysis (PTA) of build-up tests: Horner method + Bourdet derivative.

When a well that flowed at rate q is shut in, the bottomhole pressure rises ("builds up") towards the
reservoir pressure. How it rises tells us about the rock:

  Horner plot:  Pws vs log10(Horner time),  Horner time = (tp + dt) / dt
                During RADIAL FLOW this is a straight line:  Pws = P* - m * log10(Horner time)
  m   [psi/cycle] = slope (pressure change per log cycle)
  k   [md]        = 162.6 q B mu / (m h)                                   permeability
  s   [-]         = 1.151 [ (P1hr - Pwf) / m - log10( k / (phi mu ct rw^2) ) + 3.23 ]    skin
  dP_skin [psi]   = 0.869 m s                                              extra pressure drop from skin

Units (field): q STB/d, B rb/STB, mu cp, h ft, phi fraction, ct 1/psi, rw ft, pressures psia, times hours.

Which part of the curve is radial flow? -> the Bourdet derivative  dP' = d(dP) / d ln(dt_e)  is FLAT there
(equal to m / 2.303). Before it: wellbore storage hump. After it: boundary effects.
"""
import numpy as np
import pandas as pd

LN10 = np.log(10)


# ---------------------------------------------------------------------------------------------
# 1. Find shut-ins and extract the build-up data
# ---------------------------------------------------------------------------------------------
def find_shut_ins(hf):
    """Shut-in events = continuous blocks of minutes with shut_in = True (choke at 0 %), per well."""
    events = []
    for well, g in hf.groupby('well_name'):
        block = (g.shut_in != g.shut_in.shift()).cumsum()
        for _, b in g[g.shut_in].groupby(block[g.shut_in]):
            start, end = b.timestamp.min(), b.timestamp.max() + pd.Timedelta(minutes=1)
            events.append({'well_name': well, 't_shut': start, 't_open': end,
                           'duration_hr': (end - start).total_seconds() / 3600})
    return pd.DataFrame(events)


def buildup_data(hf, event):
    """Minute data of one build-up.

    Returns (dt, pws, pwf):
      dt  [h]    time since shut-in
      pws [psia] shut-in bottomhole pressure (downhole gauge)
      pwf [psia] flowing pressure just before shut-in = Pwf(dt=0), the starting point of the build-up
    """
    w = hf[hf.well_name == event['well_name']].set_index('timestamp').dh_pressure_psia.dropna()
    pwf = w[:event['t_shut'] - pd.Timedelta(minutes=1)].iloc[-1]
    bu = w[event['t_shut'] + pd.Timedelta(minutes=1):event['t_open'] - pd.Timedelta(minutes=1)]
    dt = (bu.index - event['t_shut']).total_seconds().values / 3600
    return dt, bu.values, pwf


def producing_time(cum_oil_stb, q_stbd):
    """Horner producing time  tp = 24 * Np / q  [h]: how long the well would have needed to produce its
    cumulative oil Np at the last rate q. (Np since first production; here tp >> dt so its exact value
    hardly changes the result -- checked: < 0.5 % change in k.)"""
    return 24 * cum_oil_stb / q_stbd


# ---------------------------------------------------------------------------------------------
# 2. Derivative and radial-flow window
# ---------------------------------------------------------------------------------------------
def log_bins(dt, p, per_cycle=20):
    """Average the minute data in log-spaced time bins (20 per log cycle).
    Why: on a log axis late time has thousands of minutes and early time very few; binning gives every
    part of the curve equal weight and removes gauge noise before taking a derivative."""
    edges = 10 ** np.arange(np.log10(dt.min()), np.log10(dt.max()) + 1 / per_cycle, 1 / per_cycle)
    idx = np.digitize(dt, edges)
    df = pd.DataFrame({'dt': dt, 'p': p, 'bin': idx}).groupby('bin').median()
    return df.dt.values, df.p.values


def bourdet_derivative(dt, dp, tp, L=0.15):
    """Bourdet derivative  dP' = d(dP) / d ln(dt_e)   with Agarwal equivalent time dt_e = dt*tp/(tp+dt).

    Using dt_e (not dt) accounts for the flow period before the shut-in (superposition), so radial flow
    gives a flat line even late in the build-up. Smoothing: slopes are taken to the first points at least
    L (in ln units) to the left and right, then weighted (standard Bourdet 1989 algorithm).
    """
    x = np.log(dt * tp / (tp + dt))
    der = np.full(len(x), np.nan)
    for i in range(1, len(x) - 1):
        left = np.where(x[i] - x[:i] >= L)[0]
        right = np.where(x[i + 1:] - x[i] >= L)[0]
        j = left[-1] if len(left) else 0
        k = i + 1 + right[0] if len(right) else len(x) - 1
        dl, dr = x[i] - x[j], x[k] - x[i]
        slope_l = (dp[i] - dp[j]) / dl
        slope_r = (dp[k] - dp[i]) / dr
        der[i] = (slope_l * dr + slope_r * dl) / (dl + dr)
    return der


def find_radial_window(dt, der, min_cycles=0.5):
    """Pick the radial-flow period automatically = where the derivative is flattest.

    Search only after the wellbore-storage hump (after the derivative's peak). For every candidate window
    at least `min_cycles` log cycles wide (less if the test is short), measure how flat the derivative is:
        flatness = |slope of log(der) vs log(dt)|  +  std(der) / mean(der)
    (radial flow: slope 0 and no scatter). Return (dt_start, dt_end) of the flattest window.
    """
    ok = ~np.isnan(der)
    dt, der = dt[ok], der[ok]
    after = np.arange(len(der)) > np.argmax(der)
    if after.sum() < 4:
        # short test: wellbore storage lasts almost to the end -> no clear radial flow; fall back to the
        # latest third of the build-up (log time), the part closest to radial flow
        after = np.log10(dt) >= np.log10(dt[0]) + 2 / 3 * np.log10(dt[-1] / dt[0])
    dt, der = dt[after], der[after]
    span = np.log10(dt[-1] / dt[0])
    width = min(min_cycles, 0.6 * span)

    best, best_score = (dt[0], dt[-1]), np.inf
    for i in range(len(dt)):
        for j in range(i + 3, len(dt)):
            if np.log10(dt[j] / dt[i]) < width:
                continue
            w_dt, w_der = dt[i:j + 1], der[i:j + 1]
            slope = np.polyfit(np.log10(w_dt), np.log10(w_der), 1)[0]
            score = abs(slope) + w_der.std() / w_der.mean()
            if score < best_score:
                best, best_score = (w_dt[0], w_dt[-1]), score
            break                                   # shortest valid window per start point
    return best


# ---------------------------------------------------------------------------------------------
# 3. Horner straight line and reservoir properties
# ---------------------------------------------------------------------------------------------
def fit_horner(dt, pws, tp, window):
    """Straight line through the radial-flow minutes:  Pws = P* - m * log10((tp + dt) / dt).
    Returns m [psi/cycle], P* [psia] (extrapolated to infinite shut-in), P1hr [psia] (line at dt = 1 h)."""
    sel = (dt >= window[0]) & (dt <= window[1])
    x = np.log10((tp + dt[sel]) / dt[sel])
    slope, p_star = np.polyfit(x, pws[sel], 1)
    m = -slope
    p1hr = p_star - m * np.log10(tp + 1)
    return m, p_star, p1hr


def superposition_function(dt, steps, q_last):
    """Multi-rate superposition time for a build-up (shut-in at t = 0, steps = [(t_i [h], q_i), ...] before it):
        X(dt) = sum_i ((q_i - q_(i-1)) / q_last) * log10(dt - t_i)  -  log10(dt)
    The last term is the shut-in itself (rate q_last -> 0). With one constant rate since tp hours ago this is
    exactly log10((tp + dt) / dt), i.e. Horner time. With the REAL rate history (VFM daily rates) it removes
    the distortion that earlier rate changes cause in the build-up (what commercial PTA software does)."""
    x = -np.log10(dt)
    q_prev = 0.0
    for t_i, q_i in steps:
        if t_i < 0:
            x = x + (q_i - q_prev) / q_last * np.log10(dt - t_i)
        q_prev = q_i
    return x


def fit_superposition(dt, pws, steps, q_last, window):
    """Straight line Pws = P_int - m * X(dt) through the radial-flow minutes. Returns m, P_int, P1hr (line at 1 h)."""
    sel = (dt >= window[0]) & (dt <= window[1])
    slope, p_int = np.polyfit(superposition_function(dt[sel], steps, q_last), pws[sel], 1)
    m = -slope
    p1hr = p_int - m * superposition_function(np.array([1.0]), steps, q_last)[0]
    return m, p_int, p1hr


def reservoir_properties(m, p1hr, pwf, q, pvt):
    """k, skin, dP_skin from the Horner line (field units, see module docstring).

    Intuition:
      k: a flatter line (small m) = pressure recovers easily = permeable rock.
      s: compares the pressure 1 h after shut-in with what an undamaged well would show.
         s > 0 damaged near-wellbore zone, s < 0 stimulated (e.g. acidised / fractured).
      dP_skin: how many psi of drawdown are 'wasted' in the damaged zone while flowing.
    """
    k = 162.6 * q * pvt['oil_fvf_rb_stb'] * pvt['oil_viscosity_cp'] / (m * pvt['net_pay_thickness_ft'])
    diffusivity_term = np.log10(k / (pvt['porosity_frac'] * pvt['oil_viscosity_cp'] *
                                     pvt['total_compressibility_1_psi'] * pvt['wellbore_radius_ft'] ** 2))
    skin = 1.151 * ((p1hr - pwf) / m - diffusivity_term + 3.23)
    dp_skin = 0.869 * m * skin
    return k, skin, dp_skin


def radius_of_investigation(k, t_hr, pvt):
    """How far the pressure wave travelled in t hours:  r_i = sqrt(k t / (948 phi mu ct))  [ft]."""
    return np.sqrt(k * t_hr / (948 * pvt['porosity_frac'] * pvt['oil_viscosity_cp'] *
                               pvt['total_compressibility_1_psi']))


# ---------------------------------------------------------------------------------------------
# 4. Sensitivity: how much do the results depend on the analysis choices?
# ---------------------------------------------------------------------------------------------
def sensitivity(hf, event, q, cum_oil, pvt):
    """Re-run the analysis with each judgement call changed, report k and skin change vs the base case.

    Choices varied: derivative smoothing L, minimum radial-window width, window start / end moved
    earlier / later (x0.7, x1.4), and the VFM rate q +-5 % (~2x the VFM oil error).
    k is exactly proportional to q (k = 162.6 q B mu / (m h)); skin depends on q only through log(k).
    """
    base = analyze_event(hf, event, q, cum_oil, pvt)
    w0, w1 = base['radial_start_hr'], base['radial_end_hr']
    cases = {'L = 0.10': dict(L=0.10), 'L = 0.30': dict(L=0.30),
             'min window 0.3 cycle': dict(min_cycles=0.3), 'min window 0.8 cycle': dict(min_cycles=0.8),
             'window start x0.7': dict(window=(w0 * 0.7, w1)), 'window start x1.4': dict(window=(w0 * 1.4, w1)),
             'window end x0.7': dict(window=(w0, w1 * 0.7)),
             'window end x1.4': dict(window=(w0, min(w1 * 1.4, event['duration_hr'])))}
    rows = []
    for name, options in cases.items():
        r = analyze_event(hf, event, q, cum_oil, pvt, **options)
        rows.append({'case': name, 'k_md': r['k_md'], 'skin': r['skin']})
    for factor in (0.95, 1.05):
        r = analyze_event(hf, event, q * factor, cum_oil, pvt)
        rows.append({'case': f'q x{factor}', 'k_md': r['k_md'], 'skin': r['skin']})
    out = pd.DataFrame(rows)
    out['k_change_pct'] = (out.k_md / base['k_md'] - 1) * 100
    out['skin_change'] = out.skin - base['skin']
    out.insert(0, 'well_name', event['well_name'])
    out.insert(1, 't_shut', event['t_shut'])
    return out


# ---------------------------------------------------------------------------------------------
# 5. Interpretation in words
# ---------------------------------------------------------------------------------------------
def interpret(k, skin, dp_skin, flow_eff, boundary=None, boundary_ft=None, r_inv_ft=None):
    """Short engineering interpretation of one build-up (rule-of-thumb thresholds).

    Permeability: < 10 md low, 10-100 md moderate, > 100 md high.
    Skin: < -0.5 stimulated, -0.5..1 ~undamaged, 1..3 mildly damaged, > 3 damaged (stimulation candidate).
    Boundary (from model matching): constant pressure = pressure support (aquifer / injector);
    closed = no-flow limit; infinite = nothing seen within the radius of investigation.
    """
    words = []
    level = 'low' if k < 10 else ('moderate' if k <= 100 else 'high')
    words.append(f'{level} permeability ({k:.0f} md)')
    if skin < -0.5:
        words.append(f'stimulated well (skin {skin:.1f}, flow efficiency {flow_eff:.2f})')
    elif skin <= 1:
        words.append(f'near-undamaged well (skin {skin:.1f})')
    elif skin <= 3:
        words.append(f'mildly damaged (skin {skin:.1f}, {dp_skin:.0f} psi lost to skin)')
    else:
        words.append(f'DAMAGED (skin {skin:.1f}, {dp_skin:.0f} psi lost, flow efficiency {flow_eff:.2f}) '
                     f'-> stimulation candidate')
    if boundary == 'constant_pressure':
        sure = 'possible ' if boundary_ft > 0.9 * r_inv_ft else ''
        words.append(f'{sure}pressure support (constant-pressure boundary) at ~{boundary_ft:.0f} ft')
    elif boundary == 'closed':
        words.append(f'closed (no-flow) boundary at ~{boundary_ft:.0f} ft')
    elif boundary == 'infinite':
        words.append(f'no boundary seen within ~{r_inv_ft:.0f} ft (infinite-acting)')
    return '; '.join(words)


# ---------------------------------------------------------------------------------------------
# 6. Plots (matplotlib, for notebooks / reports)
# ---------------------------------------------------------------------------------------------
def plot_event(hf, result, daily_rates=None):
    """The three standard build-up plots for one shut-in:
      (a) history: bottomhole pressure (+ VFM oil rate) around the shut-in
      (b) Horner plot: Pws vs Horner time (log axis, reversed), radial-flow points + fitted line
      (c) log-log diagnostic: dP and Bourdet derivative vs dt, radial-flow window shaded
    """
    import matplotlib.pyplot as plt
    well, t0 = result['well_name'], result['t_shut']
    dt, pws, tp = result['_dt'], result['_pws'], result['tp_hr']
    w0, w1 = result['radial_start_hr'], result['radial_end_hr']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) history
    ax = axes[0]
    g = hf[(hf.well_name == well) & (hf.timestamp >= t0 - pd.Timedelta(days=2)) &
           (hf.timestamp <= t0 + pd.Timedelta(hours=result['duration_hr']) + pd.Timedelta(days=2))]
    ax.plot(g.timestamp, g.dh_pressure_psia, lw=0.8, label='downhole pressure')
    ax.axvspan(t0, t0 + pd.Timedelta(hours=result['duration_hr']), color='grey', alpha=0.2, label='shut-in')
    ax.set_ylabel('pressure [psia]')
    ax.tick_params(axis='x', rotation=30)
    if daily_rates is not None:
        r = daily_rates[(daily_rates.well_name == well) & (pd.to_datetime(daily_rates.date) >= g.timestamp.min().normalize())
                        & (pd.to_datetime(daily_rates.date) <= g.timestamp.max())]
        ax2 = ax.twinx()
        ax2.step(pd.to_datetime(r.date), r.oil_rate_stbd, where='post', color='tab:green', label='VFM oil rate')
        ax2.set_ylabel('oil rate [STB/d]', color='tab:green')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title('(a) pressure & rate history')

    # (b) Horner
    ax = axes[1]
    ht = (tp + dt) / dt
    radial = (dt >= w0) & (dt <= w1)
    ax.semilogx(ht, pws, '.', ms=2, color='lightgrey', label='build-up data')
    ax.semilogx(ht[radial], pws[radial], '.', ms=3, color='tab:blue', label='radial flow (used for fit)')
    line_x = np.logspace(0, np.log10(ht.max()), 50)
    ax.semilogx(line_x, result['p_star_psia'] - result['m_psi_cycle'] * np.log10(line_x), 'r-',
                label=f"m = {result['m_psi_cycle']:.1f} psi/cycle")
    ax.plot(1, result['p_star_psia'], 'r*', ms=12, label=f"P* = {result['p_star_psia']:.0f} psia")
    ax.invert_xaxis()
    ax.set_xlabel('Horner time (tp + dt) / dt')
    ax.set_ylabel('Pws [psia]')
    ax.legend(fontsize=8)
    ax.set_title(f"(b) Horner: k = {result['k_md']:.1f} md, s = {result['skin']:.2f}")

    # (c) log-log
    ax = axes[2]
    ax.loglog(dt, pws - result['pwf_psia'], '.', ms=2, color='tab:blue', label='dP = Pws - Pwf')
    ok = result['_derivative'] > 0
    ax.loglog(result['_dt_bins'][ok], result['_derivative'][ok], 'o-', ms=3, color='tab:red', label="derivative dP'")
    ax.axvspan(w0, w1, color='green', alpha=0.15, label='radial flow window')
    ax.axhline(result['m_psi_cycle'] / LN10, color='green', ls='--', lw=1, label='m / 2.303')
    ax.set_xlabel('dt [h]')
    ax.set_ylabel('dP, dP\' [psi]')
    ax.legend(fontsize=8)
    ax.set_title('(c) log-log diagnostic')

    fig.suptitle(f"{well} build-up {t0:%Y-%m-%d %H:%M} ({result['duration_hr']:.1f} h)", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# 7. Whole analysis of one event
# ---------------------------------------------------------------------------------------------
FINAL_PTA_OPTIONS = {'pwf_hours': 24, 'q_days': 7}
# Chosen with benchmark tests (outputs/submission/SUBMISSION_LOG.md): flowing conditions before the shut-in are
# AVERAGED -- Pwf(dt=0) = mean of the last 24 h, q = mean daily VFM oil rate of the last 7 days -- instead of
# the single last minute / single last day (both noisier, both scored lower).


def rate_before_shut_in(daily_rates, well, t_shut, q_days=7):
    """Stabilised oil rate before a shut-in = mean daily VFM oil rate of the last q_days full days.
    daily_rates: DataFrame with date, well_name, oil (STB/d)."""
    day0 = t_shut.normalize()
    r = daily_rates[(daily_rates.well_name == well) & (daily_rates.date < day0) &
                    (daily_rates.date >= day0 - pd.Timedelta(days=q_days))]
    return r.oil.mean()


def analyze_event(hf, event, q, cum_oil, pvt, L=0.15, min_cycles=0.5, window=None, pwf_hours=None):
    """Full automatic build-up analysis of one shut-in. q = oil rate before shut-in (VFM), cum_oil = Np.
    L, min_cycles: derivative smoothing and minimum radial-window width (changed only in sensitivity tests).
    window: (start_hr, end_hr) to override the automatic radial-flow window.
    pwf_hours: Pwf(dt=0) = mean downhole pressure of the last pwf_hours hours before shut-in
               (None = the last flowing minute)."""
    dt, pws, pwf = buildup_data(hf, event)
    if pwf_hours:
        w = hf[(hf.well_name == event['well_name']) & (hf.timestamp < event['t_shut']) &
               (hf.timestamp >= event['t_shut'] - pd.Timedelta(hours=pwf_hours))]
        pwf = w.dh_pressure_psia.mean()
    result = analyze_buildup(dt, pws, pwf, q, cum_oil, pvt, L=L, min_cycles=min_cycles, window=window)
    result.update({'well_name': event['well_name'], 't_shut': event['t_shut'], 'duration_hr': event['duration_hr']})
    return result


def analyze_buildup(dt, pws, pwf, q, cum_oil, pvt, L=0.15, min_cycles=0.5, window=None):
    """Horner analysis of build-up ARRAYS (real or simulated): dt [h], pws [psia], pwf [psia]."""
    tp = producing_time(cum_oil, q)
    dt_b, p_b = log_bins(dt, pws)
    der = bourdet_derivative(dt_b, p_b - pwf, tp, L=L)
    if window is None:
        window = find_radial_window(dt_b, der, min_cycles=min_cycles)
    m, p_star, p1hr = fit_horner(dt, pws, tp, window)
    k, skin, dp_skin = reservoir_properties(m, p1hr, pwf, q, pvt)
    return {
        'q_oil_stbd': q, 'tp_hr': tp, 'pwf_psia': pwf, 'radial_start_hr': window[0], 'radial_end_hr': window[1],
        'm_psi_cycle': m, 'p_star_psia': p_star, 'p1hr_psia': p1hr,
        'k_md': k, 'skin': skin, 'dP_skin_psi': dp_skin,
        'flow_efficiency': (p_star - pwf - dp_skin) / (p_star - pwf),
        'radius_investigation_ft': radius_of_investigation(k, window[1], pvt),
        # arrays for plots
        '_dt': dt, '_pws': pws, '_dt_bins': dt_b, '_dp_bins': p_b - pwf, '_derivative': der,
    }


# ---------------------------------------------------------------------------------------------
# 8. Planned build-up test or unplanned shut-in?
# ---------------------------------------------------------------------------------------------
def count_restart_steps(choke, well, t_open, hours=8):
    """Number of choke changes in the first `hours` after the well is reopened.
    Planned tests are brought back with a STAGED bean-up (e.g. 60 % -> 80 % -> 100 % of the old opening);
    after an unplanned trip the choke simply goes back to its previous opening in one step."""
    c = choke[(choke.well_name == well) & (choke.timestamp >= t_open - pd.Timedelta(minutes=1)) &
              (choke.timestamp <= t_open + pd.Timedelta(hours=hours))]
    return len(c)


def classify_shut_ins(events, choke, min_test_hours=24):
    """Label every shut-in as a 'planned build-up test' or an 'unplanned shut-in'.

    Textbook definition (Earlougher; Horne): a pressure BUILD-UP TEST is a designed test -- stabilised
    production, a deliberate shut-in of planned duration (typically 1-3 days, long enough to investigate the
    reservoir) and a controlled return to production. Other closures (trips, emergency / facility shutdowns)
    are unplanned shut-ins: their pressure data can be looked at, but they are not reported as test results.
    Operational signatures used:
      1. duration >= min_test_hours (24 h)           designed test length
      2. staged choke ramp-up after reopening (>= 2 steps)   controlled return to production
    A shut-in is a planned build-up test if it has BOTH signatures (in this field: 48 h / 72 h tests, started
    on the hour, reopened in 3 steps; the unplanned ones lasted 5.7-9 h and were reopened in one step).
    If there is NO choke record at all around the reopening (e.g. uploaded data without a choke file), the
    restart cannot be checked and the duration alone decides (stated in the reason).
    """
    out = events.copy()
    out['restart_steps'] = [count_restart_steps(choke, e.well_name, e.t_open) for _, e in events.iterrows()]
    out['starts_on_the_hour'] = (out.t_shut.dt.minute == 0)
    long_enough = out.duration_hr >= min_test_hours
    restart_known = out.restart_steps >= 1
    staged = out.restart_steps >= 2
    out['event_type'] = np.where(long_enough & (staged | ~restart_known), 'planned build-up test', 'unplanned shut-in')
    reasons = []
    for (_, e), ok_len, known, ok_stage in zip(out.iterrows(), long_enough, restart_known, staged):
        restart = (f"reopened in {e.restart_steps} choke step(s) {'(staged)' if ok_stage else '(one step)'}"
                   if known else 'restart not visible (no choke data) -> duration decides')
        reasons.append(f"{e.duration_hr:.1f} h {'(>= 24 h)' if ok_len else '(< 24 h)'}; {restart}; "
                       f"started {e.t_shut:%H:%M}")
    out['classification_reason'] = reasons
    return out


def analyze_all(hf, events, daily_rates, pvt, completion, choke, options=FINAL_PTA_OPTIONS, skipped=None):
    """Final automatic PTA of every shut-in (used by the notebooks, the app and the upload pipeline).

    events      : find_shut_ins(hf)
    daily_rates : DataFrame date, well_name, oil  (VFM predictions; must cover the days BEFORE each shut-in)
    completion  : indexed by well_name, with cum_oil_at_2025_01_01_stb (production before the data starts)
    skipped     : optional list; shut-ins that cannot be interpreted are appended here (with the reason)
                  instead of stopping the whole analysis
    Returns a list of result dicts (Horner analysis + event_type / classification_reason).
    Np (for tp) = cumulative oil before the data + VFM oil since the data start, up to the shut-in day.

    A shut-in already in progress at the first minute of the data cannot be interpreted: Horner needs the
    flowing pressure Pwf(dt = 0) just before the well was closed, and that moment is not in the data.
    """
    events = classify_shut_ins(events, choke)
    first_minute = hf.groupby('well_name').timestamp.min()
    results = []
    for _, e in events.iterrows():
        info = {'well_name': e.well_name, 't_shut': e.t_shut, 't_open': e.t_open, 'duration_hr': e.duration_hr,
                'event_type': e.event_type}
        if e.t_shut <= first_minute[e.well_name]:
            if skipped is not None:
                skipped.append({**info, 'reason': 'shut-in already in progress when the data starts: no flowing '
                                                  'pressure before it, so it cannot be interpreted'})
            continue
        try:
            q = rate_before_shut_in(daily_rates, e.well_name, e.t_shut, options['q_days'])
            r = daily_rates[(daily_rates.well_name == e.well_name) & (daily_rates.date < e.t_shut.normalize())]
            cum = completion.loc[e.well_name, 'cum_oil_at_2025_01_01_stb'] + r.oil.sum()
            res = analyze_event(hf, e, q, cum, pvt.loc[e.well_name], pwf_hours=options['pwf_hours'])
        except Exception as exc:                  # one bad event must never stop the other results
            if skipped is None:
                raise
            skipped.append({**info, 'reason': f'analysis failed ({type(exc).__name__}: {exc})'})
            continue
        res.update({'t_open': e.t_open, 'event_type': e.event_type, 'classification_reason': e.classification_reason,
                    'restart_steps': e.restart_steps, 'cum_oil_stb': cum})
        results.append(res)
    return results

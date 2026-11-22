"""VFM methods. Each method: method(train, predict) -> DataFrame [oil_rate_stbd, water_rate_stbd, gas_rate_mscfd].

  0. well_mean           naive baseline: average rate of that well in training
  1. time_interpolation  naive baseline: straight line between the neighbouring tests in time
  B. physics             q_oil = J (P_res(t) - Pwf),  water from WC trend,  gas from mean GOR
  A. make_ml_method(...) gradient boosting / other regressors on a chosen feature set
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from validation import TARGETS

RAW_FEATURES = ['whp_psia', 'whp_downstream_psia', 'dh_pressure_psia', 'dh_temperature_degf',
                'annulus_pressure_psia', 'gas_lift_rate_mscfd', 'choke_opening_pct', 'days']

PHYSICS_FEATURES = RAW_FEATURES + ['choke_dp', 'choke_p_ratio', 'choke_size_64', 'choke_flow_term',
                                   'gilbert_term', 'tubing_dp', 'tubing_gradient', 'lift_dp',
                                   'temp_below_res', 'J_darcy', 'J_x_pwf']


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------
def well_mean(train, predict):
    """Predict the well's average rate from the training tests. Any real model must beat this."""
    means = train.groupby('well_name')[TARGETS].mean()
    return means.loc[predict.well_name].set_index(predict.index)


def time_interpolation(train, predict):
    """Straight line in time between the previous and next test of the same well
    (before the first / after the last test: hold the nearest value). Uses no sensors at all."""
    out = pd.DataFrame(index=predict.index, columns=TARGETS, dtype=float)
    for i, row in predict.iterrows():
        g = train[train.well_name == row.well_name].sort_values('days')
        for t in TARGETS:
            out.loc[i, t] = np.interp(row.days, g.days, g[t])
    return out


# ---------------------------------------------------------------------------------------------
# B. Physics model
# ---------------------------------------------------------------------------------------------
def fit_well_trends(train, pres_deg=1, j_source='darcy', wc_model='wc_linear', gor_model='mean',
                    weight_by_hours=False, wc_correction=None, pres_shrink=0.0, recency_tau=None,
                    gor_prior_weight=4, wc_column='water_cut_pct', pres_local_bw=None,
                    wc_local_bw=None, j_shrink=0.0):
    """Per-well trends fitted on TRAINING tests only. Options exist so each choice can be ablated.

    Forecasting options (matter when predicting future months, see notebooks/15):
      pres_shrink (0..1): pull each well's P_res decline slope toward the field median slope:
                          slope = (1 - s) * own slope + s * field median.  A slope from 5-9 noisy tests
                          is uncertain; wells in one reservoir deplete at broadly similar rates.
      recency_tau [days]: weight tests by exp((t - t_last) / tau) -> recent tests count more, so the trend
                          follows the latest behaviour (e.g. a decline that is speeding up).
      pres_local_bw [days]: LOCAL reservoir-pressure trend instead of one straight line per well.
                          For every day t that is predicted, a straight line is fitted to the tests with
                          Gaussian weights  w_i = exp(-0.5 * ((t_i - t) / bw)^2)  and evaluated at t.
                          Between tests it follows local changes (e.g. a decline speeding up); beyond the last
                          test the most recent tests dominate. Large bw -> the usual single straight line.
      wc_local_bw [days]: the same local trend for the water cut (instead of one straight line per well).

    Oil (inflow equation q_oil = J * (P_res - Pwf)):
      j_source='darcy' : J from Darcy (k, s from PTA). Per test P_res = Pwf + q_oil / J,
                         then a polynomial in time  P_res(t) = p0 + p1*days (+ p2*days^2 if pres_deg=2)
      j_source='fitted': J fitted from the tests themselves: regress q_oil = a + b*days - J*Pwf
                         (each test gives a rate at a different Pwf; the slope of rate vs Pwf is J).
                         Less stable with 7-9 tests, but no PTA needed.
      j_source='auto'  : Darcy J where the well has its OWN build-up (J_from_pta = True); otherwise the
                         fitted J, pulled toward the Darcy prior (field-typical k & skin):
                         J = (1 - j_shrink) * J_fitted + j_shrink * J_prior
      In every case P_res per test = Pwf + q_oil / J, then the P_res trend in time.
    Water:
      'wc_linear' / 'wc_quadratic' / 'wc_const' : water cut WC(t) as polynomial in time (deg 1 / 2 / 0)
      'wor_linear' : water-oil ratio WOR = water / oil, linear in time
      'wc_last'    : water cut of the most recent test, held constant (simple forecasting baseline)
      wc_column    : which test measurement the water-cut trend is fitted to:
                     'water_cut_pct' = 100 * water / (oil + water) from the metered rates (default)
                     'bsw_pct'       = basic sediment & water, measured on a liquid sample (shake-out / centrifuge)
    Gas (GOR = gas / oil, constant in theory because Pwf > bubble point -> no free gas in reservoir):
      'mean' / 'median' (robust to one bad test) / 'linear' (trend in time)
      'rs'        : solution GOR Rs from the completion file. Above the bubble point no gas comes out of
                    solution in the reservoir, so the produced GOR should equal Rs (no test noise at all).
      'rs_shrink' : average of the tests pulled toward Rs:
                    GOR = (n * test_mean + w * Rs) / (n + w),  n = number of tests, w = gor_prior_weight
                    (Rs counts like w extra tests -> few tests: lean on Rs, many tests: lean on data)
    weight_by_hours: longer tests average more noise away -> trust them more (weight = test hours).
    wc_correction: None, 'tubing' or 'tubing+lift'. Water is heavier than oil, so on a day with a heavier
      fluid column (higher tubing gradient, psi/ft) the water cut is likely above its trend:
        WC = WC_trend(t) + b1 * (tubing_gradient - well mean) [+ b2 * (gas_lift - well mean)]
      b1, b2 are ONE pair shared by all wells (fitted on the trend residuals) -> very little overfitting.
      Gas lift is included because more lift gas also makes the column lighter.
    """
    wc_deg = {'wc_linear': 1, 'wc_quadratic': 2, 'wc_const': 0, 'wor_linear': 1, 'wc_last': 0}[wc_model]
    rows = []
    for well, g in train.groupby('well_name'):
        weight = np.ones(len(g))
        if weight_by_hours:
            weight = weight * g.test_hours.values
        if recency_tau:
            weight = weight * np.exp((g.days.values - g.days.max()) / recency_tau)
        w = np.sqrt(weight)                                       # polyfit weights act on residuals -> sqrt
        row = {'well_name': well}

        use_darcy = j_source == 'darcy' or (j_source == 'auto' and bool(g.J_from_pta.iloc[0]))
        if use_darcy:
            J = g.J_darcy.iloc[0]
        else:
            X = np.column_stack([np.ones(len(g)), g.days, g.dh_pressure_psia])
            a, b, c = np.linalg.lstsq(X, g.oil_rate_stbd, rcond=None)[0]
            J = -c                                            # q = J*(p0 + p1*t) - J*Pwf
            if j_source == 'auto':
                J = (1 - j_shrink) * J + j_shrink * g.J_darcy.iloc[0]
            if not J > 0:                                     # nonsense fit (noise) -> fall back to the prior
                J = g.J_darcy.iloc[0]
        p_res = g.dh_pressure_psia + g.oil_rate_stbd / J
        row['pres_coef'] = np.polyfit(g.days, p_res, pres_deg, w=w)
        row['_points'] = (g.days.values, p_res.values, weight)       # kept for the slope shrinkage
        row['_pres_obs'] = (g.days.values, p_res.values)              # kept for the local trend
        row['J'] = J

        if wc_model == 'wor_linear':
            row['wc_coef'] = np.polyfit(g.days, g.water_rate_stbd / g.oil_rate_stbd, 1, w=w)
        elif wc_model == 'wc_last':                              # hold the most recent test's water cut
            row['wc_coef'] = np.array([g.sort_values('days')[wc_column].iloc[-1]])
        else:
            row['wc_coef'] = np.polyfit(g.days, g[wc_column], wc_deg, w=w)
        row['wc_model'] = wc_model
        row['_wc_obs'] = (g.days.values, g[wc_column].values)             # kept for the local trend

        rs = g.solution_gor_at_pb_scf_stb.iloc[0]          # solution GOR from the completion file
        if gor_model == 'mean':
            row['gor_coef'] = np.array([g.gor_scf_stb.mean()])
        elif gor_model == 'rs':
            row['gor_coef'] = np.array([rs])
        elif gor_model == 'rs_shrink':                      # n tests count n times, Rs counts gor_prior_weight times
            n = len(g)
            row['gor_coef'] = np.array([(n * g.gor_scf_stb.mean() + gor_prior_weight * rs) / (n + gor_prior_weight)])
        elif gor_model == 'median':
            row['gor_coef'] = np.array([g.gor_scf_stb.median()])
        else:
            row['gor_coef'] = np.polyfit(g.days, g.gor_scf_stb, 1, w=w)
        row['tubing_gradient_mean'] = g.tubing_gradient.mean()
        row['gas_lift_mean'] = g.gas_lift_rate_mscfd.mean()
        rows.append(row)
    trends = pd.DataFrame(rows).set_index('well_name')

    if pres_shrink and pres_deg == 1:
        field_slope = np.median([c[0] for c in trends.pres_coef])
        for well in trends.index:
            days, p_res, weight = trends.at[well, '_points']
            slope = (1 - pres_shrink) * trends.at[well, 'pres_coef'][0] + pres_shrink * field_slope
            intercept = np.average(p_res - slope * days, weights=weight)    # best line with that slope
            trends.at[well, 'pres_coef'] = np.array([slope, intercept])

    trends['pres_local_bw'] = pres_local_bw
    trends['wc_local_bw'] = wc_local_bw
    trends['wc_b1'], trends['wc_b2'] = 0.0, 0.0
    if wc_correction:
        wc_dev = train.water_cut_pct - [np.polyval(trends.wc_coef[w], d) for w, d in zip(train.well_name, train.days)]
        tg_dev = train.tubing_gradient - train.well_name.map(trends.tubing_gradient_mean)
        gl_dev = train.gas_lift_rate_mscfd - train.well_name.map(trends.gas_lift_mean)
        X = np.column_stack([tg_dev, gl_dev]) if wc_correction == 'tubing+lift' else np.column_stack([tg_dev])
        b = np.linalg.lstsq(X, wc_dev, rcond=None)[0]
        trends['wc_b1'] = b[0]
        trends['wc_b2'] = b[1] if len(b) > 1 else 0.0
    return trends


def local_linear_trend(days_obs, values, days_new, bandwidth):
    """Kernel-weighted straight line, refitted around every requested day (see pres_local_bw)."""
    out = np.empty(len(days_new))
    for i, t in enumerate(days_new):
        w = np.exp(-0.5 * ((days_obs - t) / bandwidth) ** 2)
        out[i] = np.polyval(np.polyfit(days_obs, values, 1, w=np.sqrt(w)), t)
    return out


def physics_predict(df, trends):
    """Rates from the physics model (works for test rows AND daily rows).

    q_oil   = J * (P_res(t) - Pwf)                [STB/d]
    q_water = q_oil * WC / (100 - WC)             [STB/d]   from WC = water / (oil + water) * 100
              (or q_oil * WOR when the water-oil-ratio model is used)
    q_gas   = q_oil * GOR / 1000                  [Mscf/d]  GOR in scf/STB, /1000 -> Mscf
    """
    out = pd.DataFrame(index=df.index, columns=TARGETS + ['P_res', 'water_cut_pct', 'gor_scf_stb'], dtype=float)
    for well, g in df.groupby('well_name'):
        tr = trends.loc[well]
        if tr.pres_local_bw:
            p_res = local_linear_trend(*tr._pres_obs, g.days.values, tr.pres_local_bw)
        else:
            p_res = np.polyval(tr.pres_coef, g.days)
        oil = np.clip(tr.J * (p_res - g.dh_pressure_psia), 0, None)
        if tr.wc_model == 'wor_linear':
            water = oil * np.clip(np.polyval(tr.wc_coef, g.days), 0, None)
        else:
            if tr.wc_local_bw:
                wc_trend = local_linear_trend(*tr._wc_obs, g.days.values, tr.wc_local_bw)
            else:
                wc_trend = np.polyval(tr.wc_coef, g.days)
            wc = (wc_trend
                  + tr.wc_b1 * (g.tubing_gradient - tr.tubing_gradient_mean)
                  + tr.wc_b2 * (g.gas_lift_rate_mscfd - tr.gas_lift_mean))
            wc = np.clip(wc, 0, 99)
            water = oil * wc / (100 - wc)
        gor = np.polyval(tr.gor_coef, g.days)
        out.loc[g.index, 'oil_rate_stbd'] = oil
        out.loc[g.index, 'water_rate_stbd'] = water
        out.loc[g.index, 'gas_rate_mscfd'] = oil * gor / 1000
        out.loc[g.index, 'P_res'] = p_res
        out.loc[g.index, 'water_cut_pct'] = 100 * water / (oil + water)
        out.loc[g.index, 'gor_scf_stb'] = gor
    return out


def make_physics_method(**options):
    """Approach B with chosen options (see fit_well_trends), e.g. make_physics_method(pres_deg=2)."""
    def method(train, predict):
        return physics_predict(predict, fit_well_trends(train, **options))[TARGETS]
    return method


physics = make_physics_method()      # default: Darcy J, linear P_res, linear WC, mean GOR


# ---------------------------------------------------------------------------------------------
# A. Machine learning on the original labels
# ---------------------------------------------------------------------------------------------
def default_lgbm():
    """Small LightGBM for a tiny dataset (84 rows): shallow trees, few rows per leaf allowed."""
    return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=5,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def xgboost():
    """XGBoost: gradient boosting like LightGBM but grows trees level by level (depth-limited)."""
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=2,
                        subsample=0.8, colsample_bytree=0.8)


def catboost():
    """CatBoost: gradient boosting with symmetric trees + built-in overfitting protection.
    Often the strongest boosting model on small tabular datasets."""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=500, learning_rate=0.03, depth=4, verbose=0,
                             allow_writing_files=False)


def random_forest():
    """Random forest: average of many deep trees, each on a random resample of the rows."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)


def extra_trees():
    """Extra trees: like random forest but with random split points -> smoother, less overfit."""
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)


def gaussian_process():
    """Gaussian process: predicts by similarity to training points (smooth curve + noise term).
    Kernel = constant * RBF (smooth trend) + white noise (measurement noise). Good for tiny datasets."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    kernel = ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel()
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                                    n_restarts_optimizer=2, random_state=0))


def svr():
    """Support vector regression (RBF kernel): fits a smooth function, ignores errors smaller than epsilon."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    model = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def knn():
    """k-nearest neighbours: average rate of the 3 most similar tests (in standardised feature space)."""
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, weights='distance'))


# (ridge is defined further down in section C, so it is wrapped in a lambda to look it up at call time)
MODELS = {'Ridge': lambda: ridge(), 'LightGBM': default_lgbm, 'XGBoost': xgboost, 'CatBoost': catboost,
          'RandomForest': random_forest, 'ExtraTrees': extra_trees, 'GaussianProcess': gaussian_process,
          'SVR': svr, 'kNN': knn}


def design_matrix(df, features, wells):
    """Model inputs: chosen features + one-hot well identity (is_GL-01 ... is_GL-10)."""
    X = df[features].copy()
    for w in wells:
        X[f'is_{w}'] = (df.well_name == w).astype(float).values
    return X


def add_well_id(X_train, X_pred, train, predict, encoding):
    """Give the model the well identity (changes X_train / X_pred in place).
    'category': one column, LightGBM's native categorical type.
    'onehot'  : 10 yes/no columns is_GL-01 ... is_GL-10 (works for every model type)."""
    wells = sorted(train.well_name.unique())
    if encoding == 'category':
        X_train['well'] = pd.Categorical(train.well_name, categories=wells)
        X_pred['well'] = pd.Categorical(predict.well_name, categories=wells)
    else:
        for w in wells:
            X_train[f'is_{w}'] = (train.well_name == w).astype(float).values
            X_pred[f'is_{w}'] = (predict.well_name == w).astype(float).values


def make_ml_method(features, make_model=default_lgbm, use_well_id=True, well_encoding='category'):
    """Return a method that trains one model per target (oil, water, gas) on `features`.

    use_well_id: give the model the well identity, so it can learn each well's own level.
    well_encoding: 'category' (LightGBM native categorical) or 'onehot' (10 yes/no columns
                   is_GL-01 ... is_GL-10, works for every model type -> fair comparison).
    """
    def method(train, predict):
        cols = list(features)
        X_train, X_pred = train[cols].copy(), predict[cols].copy()
        if use_well_id:
            add_well_id(X_train, X_pred, train, predict, well_encoding)
        out = pd.DataFrame(index=predict.index)
        for t in TARGETS:
            model = make_model()
            model.fit(X_train, train[t])
            out[t] = model.predict(X_pred)
        return out
    return method


# ---------------------------------------------------------------------------------------------
# C. Hybrid: physics + machine learning
# ---------------------------------------------------------------------------------------------
def ridge():
    """Linear regression with a penalty on large coefficients (alpha chosen by internal CV).
    Features are standardised first (mean 0, std 1) so the penalty treats them equally."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def physics_out_of_fold(train, **options):
    """Physics prediction for every TRAINING test, made with that test hidden (inner leave-one-out).

    Why: fitted on a test and then predicting the same test, physics looks better than it really is.
    ML trained on those too-small errors would learn the wrong correction. Only the hidden test's own
    well needs refitting (wells are fitted independently), which keeps this fast.
    """
    out = []
    for i in train.index:
        same_well = train[train.well_name == train.at[i, 'well_name']]
        trends = fit_well_trends(same_well.drop(index=i), **options)
        out.append(physics_predict(train.loc[[i]], trends))
    return pd.concat(out)


def physics_for_train(train, out_of_fold, **options):
    if out_of_fold:
        return physics_out_of_fold(train, **options)
    return physics_predict(train, fit_well_trends(train, **options))      # in-sample (for the ablation)


def make_residual_method(features, make_model=ridge, out_of_fold=True):
    """C1. ML corrects the physics:  rate = physics_rate * (1 + correction)

    The model learns the relative physics error  y = true / physics - 1  from the features.
    A relative error (not STB/d) lets one model serve big and small wells alike.
    """
    def method(train, predict):
        phys_train = physics_for_train(train, out_of_fold)
        phys_pred = physics_predict(predict, fit_well_trends(train))
        out = pd.DataFrame(index=predict.index)
        for t in TARGETS:
            y = train[t] / phys_train[t] - 1
            model = make_model()
            model.fit(train[features], y)
            out[t] = phys_pred[t] * (1 + model.predict(predict[features]))
        return out
    return method


def make_physics_feature_method(features, make_model=default_lgbm, out_of_fold=True, well_encoding=None):
    """C2. Physics prediction is given to the ML model as one more input feature (original targets).
    well_encoding: None (no well id) or 'onehot' / 'category' (see add_well_id)."""
    def method(train, predict):
        phys_train = physics_for_train(train, out_of_fold)
        phys_pred = physics_predict(predict, fit_well_trends(train))
        out = pd.DataFrame(index=predict.index)
        for t in TARGETS:
            X_train = train[features].assign(physics=phys_train[t].values)
            X_pred = predict[features].assign(physics=phys_pred[t].values)
            if well_encoding:
                add_well_id(X_train, X_pred, train, predict, well_encoding)
            model = make_model()
            model.fit(X_train, train[t])
            out[t] = model.predict(X_pred)
        return out
    return method


def make_engineered_target_method(features, make_model=default_lgbm, label_features=True, out_of_fold=True):
    """C3. Predict engineered targets, then convert back to oil / water / gas:

        liquid = oil + water [STB/d],  WC = 100 * water / liquid [%],  GOR = 1000 * gas / oil [scf/STB]
        oil   = liquid * (1 - WC/100)
        water = liquid * WC / 100
        gas   = oil * GOR / 1000

    Why: liquid follows the choke / inflow physics, WC and GOR change slowly -> each is easier to learn
    than oil, water and gas separately (which all share the same liquid-rate swings).
    label_features: also give the model the physics outputs (oil, WC trend, mean GOR) as inputs.
    """
    targets = ['liquid_rate_stbd', 'water_cut_pct', 'gor_scf_stb']

    def method(train, predict):
        X_train, X_pred = train[features].copy(), predict[features].copy()
        if label_features:
            phys_train = physics_for_train(train, out_of_fold)
            phys_pred = physics_predict(predict, fit_well_trends(train))
            for df, phys in [(X_train, phys_train), (X_pred, phys_pred)]:
                df['phys_liquid'] = (phys.oil_rate_stbd + phys.water_rate_stbd).values
                df['phys_wc'] = phys.water_cut_pct.values
                df['phys_gor'] = phys.gor_scf_stb.values
        pred = {}
        for t in targets:
            model = make_model()
            model.fit(X_train, train[t])
            pred[t] = model.predict(X_pred)
        wc = np.clip(pred['water_cut_pct'], 0, 99)
        out = pd.DataFrame(index=predict.index)
        out['oil_rate_stbd'] = pred['liquid_rate_stbd'] * (1 - wc / 100)
        out['water_rate_stbd'] = pred['liquid_rate_stbd'] * wc / 100
        out['gas_rate_mscfd'] = out.oil_rate_stbd * pred['gor_scf_stb'] / 1000
        return out
    return method


# ---------------------------------------------------------------------------------------------
# Ensembles: combine two methods per target
# ---------------------------------------------------------------------------------------------
def make_blend_method(method_a, method_b, weights):
    """rate = (1 - w) * A + w * B, with its own weight w per target.

    w = 0 -> only A,  w = 1 -> only B,  w = 0.5 -> plain average.
    Averaging two models whose errors are not identical cancels part of each one's error.
    Example: make_blend_method(physics, gp, {'oil_rate_stbd': 0, 'water_rate_stbd': 0.5, 'gas_rate_mscfd': 0})
    """
    def method(train, predict):
        a = method_a(train, predict)
        b = method_b(train, predict)
        out = pd.DataFrame(index=predict.index)
        for t in TARGETS:
            out[t] = (1 - weights[t]) * a[t] + weights[t] * b[t]
        return out
    return method


# ---------------------------------------------------------------------------------------------
# FINAL VFM MODEL = benchmark submission v5 (21.089 / 30): physics + GP water, gas from the solution GOR,
#                   local reservoir-pressure trend, J from OUR OWN build-up analysis (fully self-contained).
#                   Chosen by ablation notebooks 05-10, time-forward test 15, official-metric re-evaluation 16.
#   oil   = physics      q_oil = J (P_res(t) - Pwf)
#                        J: Darcy with our own build-up k & skin (6 wells) / fitted from tests (4 wells, 25 % to prior)
#                        P_res(t): LOCAL linear trend per well (Gaussian time weights, bandwidth 90 days)
#   gas   = physics      q_gas = q_oil * Rs / 1000     Rs = solution GOR from the completion file
#                        (Pwf > bubble point -> produced GOR = Rs; measured test GOR is noisier)
#   water = 0.5 * physics + 0.5 * GP      physics: q_oil * WC(t) / (100 - WC(t)), WC linear in time,
#                        recent tests weighted more (recency_tau = 120 days)
#   GP = Gaussian process on SEPARATOR-SAFE features + one-hot well id
#
# Test-separator effect: during a well test the flow goes to the test separator, which runs at lower
# pressure -> flowline (downstream) pressure is 17-29 psi lower than on normal days, WHP and Pwf drop a bit,
# and the well flows slightly more. Physics handles this through Pwf. A data-driven model trained only on
# tests, given flowline / choke pressures, extrapolates on normal days and over-predicts oil by ~10 %
# (confirmed by the PTA reference rates, which match physics within 1 %). So the GP only gets inputs
# that do not depend on where the flow is routed.
# Accuracy: see outputs/final_model_metrics.json (written by notebooks/11_vfm_deliverable.ipynb)
# ---------------------------------------------------------------------------------------------
SEPARATOR_SAFE_FEATURES = ['dh_pressure_psia', 'dh_temperature_degf', 'annulus_pressure_psia',
                           'gas_lift_rate_mscfd', 'days', 'J_darcy', 'J_x_pwf', 'temp_below_res']
FINAL_PHYSICS_OPTIONS = {'recency_tau': 120, 'gor_model': 'rs', 'pres_local_bw': 90,
                         'j_source': 'auto', 'j_shrink': 0.25}   # J: own PTA where available, else fitted (25 % toward prior)
FINAL_GP_WEIGHTS = {'oil_rate_stbd': 0.0, 'water_rate_stbd': 0.5, 'gas_rate_mscfd': 0.0}
FINAL_GP_FEATURES = SEPARATOR_SAFE_FEATURES


def final_method():
    """The final model as a validation method(train, predict) -> used in notebooks to score it."""
    gp = make_ml_method(FINAL_GP_FEATURES, gaussian_process, use_well_id=True, well_encoding='onehot')
    return make_blend_method(make_physics_method(**FINAL_PHYSICS_OPTIONS), gp, FINAL_GP_WEIGHTS)


def train_final_model(test_table, physics_options=None, name='Final VFM (v5): physics + Gaussian process (water)'):
    """Fit the final model on ALL well tests. Returns a dict that can be saved (joblib / pickle).
    physics_options: override FINAL_PHYSICS_OPTIONS (used to build experimental submission versions)."""
    options = FINAL_PHYSICS_OPTIONS if physics_options is None else physics_options
    wells = sorted(test_table.well_name.unique())
    X = design_matrix(test_table, FINAL_GP_FEATURES, wells)
    gps = {}
    for target, weight in FINAL_GP_WEIGHTS.items():
        if weight > 0:
            gps[target] = gaussian_process().fit(X, test_table[target])
    return {'name': name, 'trends': fit_well_trends(test_table, **options),
            'gp': gps, 'gp_weights': FINAL_GP_WEIGHTS, 'wells': wells, 'features': FINAL_GP_FEATURES,
            'physics_options': options, 'trained_until': str(test_table.time.max())}


def predict_rates(model, table):
    """Daily (or test-window) rates from the final model.

    1. Rates while flowing: physics, blended with the Gaussian process where its weight > 0:
           rate = (1 - w) * physics + w * GP
    2. Daily rate = flowing rate x on_stream_frac   (e.g. shut in 8 of 24 h -> x 0.667)
    3. Fully shut-in days (no flowing minutes -> no features) -> 0.
    """
    flowing = table[table.dh_pressure_psia.notna()]
    phys = physics_predict(flowing, model['trends'])
    X = design_matrix(flowing, model['features'], model['wells'])

    rates = phys[TARGETS].copy()
    for target, gp in model['gp'].items():
        w = model['gp_weights'][target]
        rates[target] = (1 - w) * phys[target] + w * gp.predict(X)
    rates = rates.clip(lower=0)

    rates = rates.reindex(table.index).fillna(0.0)
    return rates.mul(table.on_stream_frac, axis=0)

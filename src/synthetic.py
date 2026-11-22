"""Approach D: synthetic data from the calibrated physics -> pretrain ML -> fine-tune on real tests.

Simulator (inside every validation fold, using TRAINING tests only -> no leakage):
  inputs : the real daily sensor rows (realistic combinations of pressures, choke, gas lift, temperature),
           copied and jittered by ~1 % so the model also sees nearby operating points
           (a 1 % change in Pwf ~ 30 psi -> the inflow equation gives a different rate -> the model can
           learn how rate responds to Pwf)
  labels : physics model  q_oil = J (P_res(t) - Pwf),  water from WC trend,  gas from GOR
           + random measurement noise of the size we measured (oil ~2.5 %, water ~4 %, gas ~6 %)

Training modes:
  real_only      : 84 real tests only (reference)
  synthetic_only : synthetic only -> can ML learn the physics from examples?
  finetune       : train on synthetic, then CONTINUE training on the real tests
                   (boosting: add extra trees fitted to the real data; neural net: extra epochs on real data)
  mixed          : synthetic + real in one training set, real rows repeated so both count equally
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import vfm
from features import SENSORS, add_physics_features
from validation import TARGETS

NOISE_PCT = {'oil_rate_stbd': 2.5, 'water_rate_stbd': 4.0, 'gas_rate_mscfd': 6.0}
RAW_COLUMNS = ['well_name', 'time'] + SENSORS + ['choke_opening_pct']
FINETUNE_ROUNDS = 100        # extra boosting trees fitted on the real tests
FINETUNE_EPOCHS = 100        # extra neural-network passes over the real tests


# ---------------------------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------------------------
def make_synthetic(daily, wells, trends, copies=1, jitter_pct=1.0, n_max=None, seed=0):
    """Synthetic training rows: real daily inputs (jittered) + physics labels + noise."""
    rng = np.random.default_rng(seed)
    flowing = daily[(daily.on_stream_frac > 0) & daily.dh_pressure_psia.notna()][RAW_COLUMNS]

    parts = []
    for _ in range(copies):
        d = flowing.copy()
        for col in SENSORS + ['choke_opening_pct']:
            d[col] = d[col] * (1 + rng.normal(0, jitter_pct / 100, len(d)))
        parts.append(d)
    syn = pd.concat(parts, ignore_index=True)
    if n_max and len(syn) > n_max:
        syn = syn.sample(n_max, random_state=seed).reset_index(drop=True)

    syn = add_physics_features(syn, wells)                 # recompute choke term, gradients ... from jittered sensors
    rates = vfm.physics_predict(syn, trends)
    for t in TARGETS:
        syn[t] = rates[t] * (1 + rng.normal(0, NOISE_PCT[t] / 100, len(syn)))
    return syn


design_matrix = vfm.design_matrix          # features + one-hot well identity (shared with the final model)


# ---------------------------------------------------------------------------------------------
# Neural network (multi-layer perceptron)
# ---------------------------------------------------------------------------------------------
def mlp():
    """Neural network: 2 hidden layers of 64 neurons. Learns by gradient descent (Adam optimiser)."""
    return MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=300,
                        alpha=1e-3, random_state=0)


def fit_predict_mlp(mode, Xs, ys, Xr, yr, Xp):
    """Neural nets need scaled inputs AND outputs (mean 0, std 1); scalers come from the first training set.
    finetune: after training on synthetic data, partial_fit runs extra passes over only the real tests,
    nudging the same weights towards the real data (this is what 'fine-tuning' means)."""
    first_X, first_y = {'real_only': (Xr, yr), 'synthetic_only': (Xs, ys), 'finetune': (Xs, ys),
                        'mixed': (pd.concat([Xs, Xr]), pd.concat([ys, yr]))}[mode]
    x_scaler = StandardScaler().fit(first_X)
    y_mean, y_std = first_y.mean(), first_y.std()

    net = mlp()
    net.fit(x_scaler.transform(first_X), (first_y - y_mean) / y_std)
    if mode == 'finetune':
        for _ in range(FINETUNE_EPOCHS):
            net.partial_fit(x_scaler.transform(Xr), (yr - y_mean) / y_std)
    return net.predict(x_scaler.transform(Xp)) * y_std + y_mean


# ---------------------------------------------------------------------------------------------
# Boosting fine-tune: continue training an existing model
# ---------------------------------------------------------------------------------------------
def finetune_boosting(model_name, Xs, ys, Xr, yr, Xp):
    """Pretrain on synthetic, then add FINETUNE_ROUNDS extra trees fitted to the real tests.
    The extra trees learn whatever the real data says that the synthetic data did not."""
    if model_name == 'LightGBM':
        from lightgbm import LGBMRegressor
        base = vfm.default_lgbm().fit(Xs, ys)
        tuned = LGBMRegressor(n_estimators=FINETUNE_ROUNDS, learning_rate=0.03, num_leaves=7,
                              min_child_samples=5, verbose=-1)
        tuned.fit(Xr, yr, init_model=base.booster_)
    elif model_name == 'XGBoost':
        from xgboost import XGBRegressor
        base = vfm.xgboost().fit(Xs, ys)
        tuned = XGBRegressor(n_estimators=FINETUNE_ROUNDS, learning_rate=0.03, max_depth=3)
        tuned.fit(Xr, yr, xgb_model=base.get_booster())
    else:  # CatBoost
        from catboost import CatBoostRegressor
        base = vfm.catboost().fit(Xs, ys)
        tuned = CatBoostRegressor(iterations=FINETUNE_ROUNDS, learning_rate=0.03, depth=4, verbose=0,
                                  allow_writing_files=False)
        tuned.fit(Xr, yr, init_model=base)
    return tuned.predict(Xp)


# ---------------------------------------------------------------------------------------------
# One method for every (model, mode) combination
# ---------------------------------------------------------------------------------------------
FINETUNABLE = ['LightGBM', 'XGBoost', 'CatBoost', 'MLP']
GP_MAX_SYNTHETIC = 300       # Gaussian process cost grows with n^3 -> use a subsample of the synthetic rows


def make_synthetic_method(model_name, mode, daily, wells, features=vfm.PHYSICS_FEATURES, copies=1):
    """Return a validation method(train, predict) for one model type and one training mode."""
    def method(train, predict):
        well_list = sorted(train.well_name.unique())
        Xr = design_matrix(train, features, well_list)
        Xp = design_matrix(predict, features, well_list)

        syn = None
        if mode != 'real_only':
            trends = vfm.fit_well_trends(train)                        # simulator calibrated on train only
            n_max = GP_MAX_SYNTHETIC if model_name == 'GaussianProcess' else None
            syn = make_synthetic(daily, wells, trends, copies=copies, n_max=n_max)
            Xs = design_matrix(syn, features, well_list)

        out = pd.DataFrame(index=predict.index)
        for t in TARGETS:
            yr = train[t]
            ys = syn[t] if syn is not None else None
            if model_name == 'MLP':
                out[t] = fit_predict_mlp(mode, Xs if syn is not None else None, ys, Xr, yr, Xp)
            elif mode == 'finetune':
                out[t] = finetune_boosting(model_name, Xs, ys, Xr, yr, Xp)
            else:
                if mode == 'real_only':
                    X_fit, y_fit = Xr, yr
                elif mode == 'synthetic_only':
                    X_fit, y_fit = Xs, ys
                else:  # mixed: repeat real rows so real and synthetic carry equal total weight
                    k = max(1, round(len(Xs) / len(Xr)))
                    X_fit = pd.concat([Xs] + [Xr] * k)
                    y_fit = pd.concat([ys] + [yr] * k)
                model = vfm.MODELS[model_name]()
                model.fit(X_fit, y_fit)
                out[t] = model.predict(Xp)
        return out
    return method

# WellMind — Experiment log

Every experiment behind the final model, with its numbers. The method and the reasoning are in [METHODOLOGY.md](METHODOLOGY.md).
- **Where the numbers come from:** the tables below are generated from the result files in [`../outputs/`](../outputs/).
- **The code behind them:** in [`../notebooks/research/`](../notebooks/research/) (one notebook per study).

**How to read the experiment names**

| Code | Family |
|---|---|
| `0` | Naive baselines (repeat the last test, well mean, time interpolation) |
| `A` | Pure machine learning on *raw* sensor features or on *physics* features (drawdown, J, choke and tubing terms) |
| `B` | Physics model per well (inflow law + trends) and its component ablations |
| `C1` | Hybrid: ML learns the residual of physics |
| `C2` | Hybrid: physics prediction used as an ML feature |
| `C3` | Engineered targets: predict liquid rate, water cut and GOR instead of oil / water / gas |
| `D` | Sim-to-real: ML pre-trained on a calibrated simulator (`synthetic_only`, `mixed`, `finetune` on real tests) |
| `E`, `F` | Ensembles / blends of physics and Gaussian process (GP) |
| `OOF` | Label-based features computed out-of-fold (no leakage) |
| `rec120` | Recency weighting of tests, τ = 120 days |
| `GP-safe` | GP restricted to separator-safe inputs (no flowline / choke pressures) |

**Metrics**
- **WAPE** = Σ|pred − true| / Σ|true| (volume-weighted, as in the official score).
- **MAPE** = mean |error| / true.
- **Between tests** = leave-one-test-out (85 folds).
- **Forecast** = trained on tests before 1 Jun or 1 Jul, scored on the next 92 days (56 forecasts).
- **Normal-day bias** = median % difference from the physics model on ~2,700 normal flowing days, where no labels exist. It measures the test-separator trap (METHODOLOGY section 8).

---

## 1. Official-metric evaluation of 47 experiments

All experiments use the same folds, scored with WAPE in both schemes, plus the normal-day check. Sorted by forecast WAPE (mean of oil, water, gas).
- **Selected family:** `F2 rec120 + GP-safe water`. The final model adds gas = oil × Rs, the local reservoir-pressure trend and own-build-up J (section 2).
- **Top of the table, rejected:** `C3 eng-targets Ridge` ranks first on forecast WAPE but under-predicts water by 16.7 % on normal days.

| experiment | between tests WAPE | forecast WAPE | fc oil | fc water | fc gas | normal-day bias oil % | normal-day bias water % |
|---|---|---|---|---|---|---|---|
| C3 eng-targets Ridge + label feats OOF | 3.22 | 4.41 | 3.17 | 4.15 | 5.91 | -3.38 | -16.67 |
| **F2 rec120 + GP-safe water (selected family)** | **3.12** | **4.49** | **3.63** | **4.95** | **4.88** | **0.00** | **0.82** |
| F1 rec120 + GP-all oil&water | 3.07 | 4.54 | 3.46 | 5.27 | 4.88 | 5.20 | 1.87 |
| rec120 + GP-safe oil&water | 3.06 | 4.58 | 3.90 | 4.95 | 4.88 | 0.40 | 0.82 |
| E3 physics + GP-all oil&water | 3.07 | 4.65 | 3.53 | 5.14 | 5.29 | 5.34 | 2.07 |
| B physics rec60 | 3.57 | 4.69 | 3.44 | 5.99 | 4.66 | 0.05 | 0.02 |
| B physics rec120 | 3.27 | 4.74 | 3.63 | 5.69 | 4.88 | 0.00 | 0.00 |
| B physics rec120 + WC tubing | 3.29 | 4.77 | 3.63 | 5.80 | 4.88 | 0.00 | -0.06 |
| E2 physics + GP-all water | 3.06 | 4.82 | 4.02 | 5.14 | 5.29 | 0.00 | 2.07 |
| D ExtraTrees synthetic_only | 4.22 | 4.87 | 3.85 | 5.15 | 5.60 | 0.18 | -0.02 |
| C1 residual Ridge OOF | 3.24 | 4.90 | 3.96 | 5.53 | 5.21 | 0.03 | 0.02 |
| B physics | 3.24 | 4.94 | 4.02 | 5.50 | 5.29 | 0.00 | -0.00 |
| B physics GOR median | 3.26 | 4.99 | 4.02 | 5.50 | 5.44 | 0.00 | -0.00 |
| B physics shrink0.3 | 3.64 | 5.07 | 4.33 | 5.48 | 5.40 | -0.00 | -0.02 |
| C2 physics-feature GP OOF | 3.26 | 5.15 | 3.61 | 5.32 | 6.50 | 10.20 | 3.37 |
| C2 physics-feature Ridge OOF | 3.42 | 5.21 | 3.76 | 5.51 | 6.36 | 3.36 | -15.78 |
| A GaussianProcess physics feats | 3.30 | 5.22 | 3.75 | 5.30 | 6.60 | 10.39 | 3.73 |
| D LightGBM mixed | 4.26 | 5.30 | 5.04 | 5.07 | 5.79 | 0.40 | 0.84 |
| C1 residual LightGBM OOF | 4.30 | 5.38 | 3.55 | 6.35 | 6.24 | 0.28 | 0.71 |
| D CatBoost mixed | 3.95 | 5.41 | 5.10 | 5.32 | 5.80 | 1.04 | 0.93 |
| D CatBoost finetune | 3.78 | 5.44 | 5.06 | 5.60 | 5.65 | 2.83 | 2.63 |
| A GaussianProcess raw feats | 3.48 | 5.81 | 3.99 | 5.69 | 7.77 | 18.93 | 13.81 |
| A GaussianProcess separator-safe feats | 3.60 | 6.10 | 5.43 | 4.65 | 8.22 | 0.80 | 1.63 |
| D XGBoost finetune | 4.23 | 6.37 | 4.66 | 8.59 | 5.86 | 3.89 | -0.37 |
| A Ridge physics feats | 3.77 | 6.38 | 5.10 | 6.40 | 7.63 | 12.18 | -3.90 |
| C3 eng-targets LightGBM + label feats OOF | 5.04 | 7.11 | 6.59 | 6.66 | 8.07 | 9.80 | 8.70 |
| D MLP finetune | 4.18 | 7.83 | 6.20 | 6.93 | 10.35 | 1.94 | -2.56 |
| A Ridge raw feats | 5.44 | 7.87 | 7.36 | 7.55 | 8.69 | 58.74 | 23.66 |
| A CatBoost physics feats | 4.96 | 7.88 | 7.30 | 6.82 | 9.54 | 20.52 | 1.06 |
| 0 naive last test value | 4.26 | 7.90 | 8.16 | 5.32 | 10.22 | 2.48 | 2.62 |
| C3 eng-targets LightGBM | 5.29 | 8.31 | 6.51 | 9.69 | 8.74 | 21.47 | 10.96 |
| C2 physics-feature LightGBM OOF | 5.94 | 8.73 | 6.70 | 10.62 | 8.86 | 17.78 | 2.66 |
| A kNN physics feats | 4.48 | 8.96 | 9.90 | 5.90 | 11.09 | 3.96 | 3.12 |
| A kNN raw feats | 4.36 | 9.02 | 9.97 | 5.99 | 11.08 | 4.79 | 2.78 |
| A CatBoost raw feats | 5.66 | 9.23 | 8.76 | 9.78 | 9.17 | 49.97 | 5.05 |
| A ExtraTrees physics feats | 4.78 | 9.24 | 9.65 | 7.02 | 11.06 | 8.54 | 2.21 |
| A ExtraTrees raw feats | 4.66 | 9.37 | 9.79 | 6.16 | 12.17 | 12.72 | 2.35 |
| A LightGBM physics feats | 6.68 | 9.50 | 6.92 | 10.81 | 10.78 | 29.62 | 2.03 |
| A RandomForest physics feats | 6.48 | 9.71 | 9.18 | 9.47 | 10.49 | 12.31 | 2.38 |
| 0 naive well mean | 7.62 | 11.63 | 12.91 | 8.44 | 13.55 | 2.57 | 2.20 |
| A LightGBM raw feats | 6.29 | 11.70 | 10.17 | 10.53 | 14.39 | 45.88 | 3.67 |
| A RandomForest raw feats | 6.78 | 12.20 | 13.10 | 10.72 | 12.78 | 18.63 | 3.47 |
| A SVR physics feats | 4.40 | 13.17 | 15.71 | 10.22 | 13.58 | 29.23 | 14.79 |
| A XGBoost physics feats | 6.80 | 13.51 | 15.89 | 5.84 | 18.80 | 48.56 | 3.54 |
| A MLP physics feats | 4.88 | 15.37 | 13.24 | 12.73 | 20.14 | 16.33 | 2.91 |
| A SVR raw feats | 4.38 | 15.64 | 18.03 | 12.66 | 16.23 | 44.84 | 22.24 |
| A XGBoost raw feats | 7.14 | 17.86 | 19.71 | 8.00 | 25.88 | 54.17 | 5.65 |

## 2. Final model, step by step

Each component of the final configuration, added one at a time (WAPE %, between tests / forecast).

| step | oil | water | gas |
|---|---|---|---|
| Repeat last test (naive) | 3.14 / 8.16 | 4.03 / 5.32 | 5.61 / 10.22 |
| Physics: J x drawdown, trends | 2.08 / 4.12 | 3.39 / 5.53 | 4.33 / 5.39 |
| + recent tests weighted | 2.16 / 3.69 | 3.50 / 5.71 | 4.32 / 4.95 |
| + gas = oil x Rs | 2.16 / 3.69 | 3.50 / 5.71 | 4.40 / 6.00 |
| + local reservoir-pressure trend | 2.03 / 3.52 | 3.38 / 5.55 | 4.33 / 5.63 |
| **+ Gaussian process for water (final)** | **2.03 / 3.52** | **3.08 / 4.92** | **4.33 / 5.63** |

Against the test labels, gas = oil × Rs looks slightly worse. The labels carry separator measurement noise, while Rs is the clean physical constant. On the organisers' benchmark (noise-free truth) it improved the score from 20.975 to 21.183.

## 3. Hypotheses tested against the organisers' benchmark

A change was kept only if time-forward validation agreed. The benchmark was used to test *how the data is generated*, not to tune parameters.

| Hypothesis | Benchmark VFM / 30 | Validation | Decision |
|---|---|---|---|
| Baseline: physics + GP water (F2) | 20.975 | — | — |
| Gas = oil × solution GOR (Rs) | **21.183** | worse on noisy test labels, physically correct | kept |
| Water cut from BSW instead of metered water cut | 21.118 | — | rejected |
| Local reservoir-pressure trend (90-day window) | **21.224** | forecast oil WAPE 3.63 → 3.48 %, p = 0.002 | kept |
| Local water-cut trend | not submitted | not significant (p = 0.28 / 0.75) | rejected |
| Temperature as extra information | not submitted | physics residual vs temperature r = −0.06 | rejected |
| Build-up pressures as reservoir-pressure anchors | not submitted | trend already within 1.2–4 psi | no gain |
| J from our own build-up analysis (self-contained, final) | **21.089** | oil WAPE 4.85 vs 5.10 % forecasting where both exist | **final** |

PTA benchmark: reporting only the three planned build-up tests scores **12.70 / 20**.

## 4. Leave-one-test-out studies (MAPE %)

The first round of studies:
- naive baselines;
- the physics model and the ablation of each of its components;
- hybrids;
- the full model zoo on raw and physics features;
- ensembles.

| experiment | oil_MAPE | water_MAPE | gas_MAPE | mean_MAPE |
|---|---|---|---|---|
| 0 well mean | 8.08 | 7.05 | 9.89 | 8.34 |
| 1 time interpolation | 4.09 | 4.67 | 7.66 | 5.47 |
| B physics | 2.40 | 3.96 | 5.74 | 4.04 |
| A LightGBM raw features | 7.66 | 9.43 | 10.08 | 9.06 |
| A LightGBM + physics features | 5.52 | 9.39 | 9.58 | 8.16 |
| B default (darcy J, P_res lin, WC lin, GOR mean) | 2.40 | 3.96 | 5.74 | 4.04 |
| B P_res constant | 8.45 | 9.20 | 11.10 | 9.58 |
| B P_res quadratic | 2.43 | 3.77 | 5.84 | 4.01 |
| B J fitted from tests | 2.61 | 3.62 | 5.77 | 4.00 |
| B WC constant | 2.40 | 10.04 | 5.74 | 6.06 |
| B WC quadratic | 2.40 | 4.80 | 5.74 | 4.31 |
| B WOR linear | 2.40 | 3.95 | 5.74 | 4.03 |
| B GOR median | 2.40 | 3.96 | 5.93 | 4.10 |
| B GOR linear | 2.40 | 3.96 | 6.59 | 4.32 |
| B weight by test hours | 2.48 | 3.98 | 5.70 | 4.05 |
| B + WC tubing correction | 2.40 | 3.83 | 5.74 | 3.99 |
| B + WC tubing+lift correction | 2.40 | 3.84 | 5.74 | 3.99 |
| C1 residual Ridge OOF | 2.40 | 3.97 | 5.75 | 4.04 |
| C1 residual Ridge in-sample | 2.40 | 3.97 | 5.75 | 4.04 |
| C1 residual LightGBM OOF | 3.05 | 5.31 | 7.84 | 5.40 |
| C2 physics-as-feature LightGBM OOF | 4.21 | 9.22 | 7.96 | 7.13 |
| C2 physics-as-feature LightGBM in-sample | 4.30 | 8.92 | 7.79 | 7.00 |
| C2 physics-as-feature Ridge OOF | 2.56 | 4.05 | 6.27 | 4.29 |
| C3 eng. targets LightGBM no label feats | 4.09 | 5.08 | 9.27 | 6.15 |
| C3 eng. targets LightGBM + label feats OOF | 3.94 | 4.95 | 8.40 | 5.77 |
| C3 eng. targets Ridge + label feats OOF | 2.60 | 3.58 | 6.36 | 4.18 |
| A Ridge raw feats | 5.64 | 6.18 | 7.59 | 6.47 |
| A Ridge physics feats | 3.40 | 4.44 | 6.99 | 4.94 |
| A LightGBM raw feats | 5.96 | 8.71 | 8.81 | 7.82 |
| A LightGBM physics feats | 5.58 | 9.58 | 9.23 | 8.13 |
| A XGBoost raw feats | 9.60 | 6.13 | 14.09 | 9.94 |
| A XGBoost physics feats | 8.48 | 5.44 | 12.38 | 8.77 |
| A CatBoost raw feats | 5.20 | 6.76 | 9.59 | 7.19 |
| A CatBoost physics feats | 4.72 | 5.18 | 8.84 | 6.24 |
| A RandomForest raw feats | 6.03 | 8.61 | 10.48 | 8.37 |
| A RandomForest physics feats | 5.70 | 7.23 | 10.38 | 7.77 |
| A ExtraTrees raw feats | 4.24 | 5.04 | 8.28 | 5.85 |
| A ExtraTrees physics feats | 4.20 | 5.57 | 8.09 | 5.95 |
| A GaussianProcess raw feats | 3.07 | 3.45 | 7.02 | 4.52 |
| A GaussianProcess physics feats | 2.81 | 3.24 | 6.71 | 4.26 |
| A SVR raw feats | 4.56 | 5.49 | 8.44 | 6.16 |
| A SVR physics feats | 4.77 | 5.29 | 8.36 | 6.14 |
| A kNN raw feats | 3.88 | 4.34 | 8.00 | 5.41 |
| A kNN physics feats | 3.91 | 4.38 | 8.14 | 5.48 |
| A CatBoost physics feats NO well id | 5.18 | 5.37 | 9.56 | 6.70 |
| B physics (baseline) | 2.40 | 3.96 | 5.74 | 4.04 |
| E1 physics oil/gas + GP water | 2.40 | 3.24 | 5.74 | 3.80 |
| E2 physics oil/gas + 50/50 water | 2.40 | 3.36 | 5.74 | 3.84 |
| E3 50/50 physics+GP all targets | 2.53 | 3.36 | 6.08 | 3.99 |
| E4 physics oil/gas + GP(+phys input) water | 2.40 | 3.27 | 5.74 | 3.80 |
| E5 physics oil/gas + 50/50 GP(+phys input) water | 2.40 | 3.40 | 5.74 | 3.85 |
| D Ridge synthetic_only | 6.75 | 13.33 | 8.21 | 9.43 |
| D Ridge mixed | 4.51 | 6.65 | 7.15 | 6.10 |
| D kNN synthetic_only | 4.96 | 5.90 | 7.96 | 6.28 |
| D kNN mixed | 4.06 | 4.72 | 9.05 | 5.94 |
| D ExtraTrees synthetic_only | 2.98 | 4.27 | 6.56 | 4.60 |
| D ExtraTrees mixed | 2.66 | 4.43 | 6.83 | 4.64 |
| D RandomForest synthetic_only | 2.92 | 5.31 | 7.07 | 5.10 |
| D RandomForest mixed | 2.98 | 4.75 | 7.39 | 5.04 |
| D GaussianProcess synthetic_only | 6.08 | 14.09 | 11.81 | 10.66 |
| D GaussianProcess mixed | 3.15 | 9.82 | 14.17 | 9.05 |
| D LightGBM synthetic_only | 3.19 | 6.32 | 6.58 | 5.36 |
| D LightGBM mixed | 3.19 | 4.83 | 7.12 | 5.05 |
| D LightGBM finetune | 2.94 | 6.21 | 6.93 | 5.36 |
| D XGBoost synthetic_only | 3.40 | 6.33 | 6.53 | 5.42 |
| D XGBoost mixed | 3.57 | 5.46 | 6.95 | 5.33 |
| D XGBoost finetune | 2.87 | 5.31 | 6.80 | 4.99 |
| D CatBoost synthetic_only | 3.97 | 5.10 | 6.51 | 5.19 |
| D CatBoost mixed | 2.94 | 4.97 | 6.30 | 4.73 |
| D CatBoost finetune | 3.09 | 4.54 | 6.27 | 4.63 |
| D MLP synthetic_only | 6.44 | 6.80 | 9.78 | 7.67 |
| D MLP mixed | 3.88 | 5.06 | 9.63 | 6.19 |
| D MLP finetune | 3.53 | 4.95 | 8.49 | 5.66 |
| D MLP real_only | 5.85 | 6.16 | 7.93 | 6.65 |

**Key readings**
- Removing the reservoir-pressure decline (`P_res constant`) or the water-cut trend (`WC constant`) breaks the model, so both physics components are essential.
- Residual learning on top of physics (`C1 residual Ridge`) learns a correction of ≈ 0: nothing systematic is left.
- Among pure ML models, smooth ones (Gaussian process, Ridge, kNN) beat every tree model.
- GP improves water significantly (paired Wilcoxon), but not oil or gas → GP only for water.

## 5. Sim-to-real: pre-training on a calibrated simulator (MAPE %)

The simulator uses real daily inputs (jittered 1 %), physics labels and realistic test noise (2.5 / 4 / 6 %), calibrated inside each fold.
- **Result:** pre-training helps tree models a lot, but none reaches the physics model.

| experiment | oil_MAPE | water_MAPE | gas_MAPE | mean_MAPE |
|---|---|---|---|---|
| D Ridge synthetic_only | 6.75 | 13.33 | 8.21 | 9.43 |
| D Ridge mixed | 4.51 | 6.65 | 7.15 | 6.10 |
| D kNN synthetic_only | 4.96 | 5.90 | 7.96 | 6.28 |
| D kNN mixed | 4.06 | 4.72 | 9.05 | 5.94 |
| D ExtraTrees synthetic_only | 2.98 | 4.27 | 6.56 | 4.60 |
| D ExtraTrees mixed | 2.66 | 4.43 | 6.83 | 4.64 |
| D RandomForest synthetic_only | 2.92 | 5.31 | 7.07 | 5.10 |
| D RandomForest mixed | 2.98 | 4.75 | 7.39 | 5.04 |
| D GaussianProcess synthetic_only | 6.08 | 14.09 | 11.81 | 10.66 |
| D GaussianProcess mixed | 3.15 | 9.82 | 14.17 | 9.05 |
| D LightGBM synthetic_only | 3.19 | 6.32 | 6.58 | 5.36 |
| D LightGBM mixed | 3.19 | 4.83 | 7.12 | 5.05 |
| D LightGBM finetune | 2.94 | 6.21 | 6.93 | 5.36 |
| D XGBoost synthetic_only | 3.40 | 6.33 | 6.53 | 5.42 |
| D XGBoost mixed | 3.57 | 5.46 | 6.95 | 5.33 |
| D XGBoost finetune | 2.87 | 5.31 | 6.80 | 4.99 |
| D CatBoost synthetic_only | 3.97 | 5.10 | 6.51 | 5.19 |
| D CatBoost mixed | 2.94 | 4.97 | 6.30 | 4.73 |
| D CatBoost finetune | 3.09 | 4.54 | 6.27 | 4.63 |
| D MLP synthetic_only | 6.44 | 6.80 | 9.78 | 7.67 |
| D MLP mixed | 3.88 | 5.06 | 9.63 | 6.19 |
| D MLP finetune | 3.53 | 4.95 | 8.49 | 5.66 |
| D MLP real_only | 5.85 | 6.16 | 7.93 | 6.65 |

## 6. Time-forward studies (MAPE %)

Training only on tests before 1 Jun or 1 Jul, then predicting the next 92 days. This is the closest test to new, later data.
- **Final configuration:** `F2 ... (FINAL)`.

| experiment | oil_MAPE | water_MAPE | gas_MAPE | forecast mean MAPE | between tests mean MAPE |
|---|---|---|---|---|---|
| naive: last test value | 9.17 | 5.94 | 11.96 | 9.02 | 5.47 |
| naive: well mean | 12.71 | 8.84 | 14.52 | 12.02 | 8.34 |
| B physics | 5.75 | 5.65 | 8.50 | 6.63 | 4.41 |
| B P_res constant | 15.77 | 16.47 | 20.02 | 17.42 | 10.29 |
| B P_res quadratic | 10.27 | 11.55 | 11.05 | 10.96 | 4.29 |
| B J fitted | 5.42 | 5.67 | 8.62 | 6.57 | 3.97 |
| B WC last value | 5.75 | 7.17 | 8.50 | 7.14 | 9.16 |
| B WC constant | 5.75 | 12.69 | 8.50 | 8.98 | 6.34 |
| B slope shrink 0.3 | 4.39 | 5.81 | 8.44 | 6.22 | 4.62 |
| B slope shrink 0.5 | 4.43 | 6.11 | 8.56 | 6.37 | 4.84 |
| B recency tau 60 d | 4.85 | 6.20 | 7.63 | 6.23 | 4.91 |
| B recency tau 120 d | 5.19 | 5.73 | 7.95 | 6.29 | 4.50 |
| A GP (all physics feats) | 4.74 | 6.17 | 8.36 | 6.43 | 4.26 |
| A GP (separator-safe feats) | 6.75 | 4.35 | 13.85 | 8.32 | 4.50 |
| A Ridge | 6.40 | 8.45 | 7.86 | 7.57 | 4.95 |
| A CatBoost | 12.63 | 7.01 | 15.24 | 11.63 | 6.42 |
| E2 physics + GP-all water (previous final) | 5.75 | 5.23 | 8.50 | 6.49 | 4.13 |
| F1 rec120 + GP-all oil & water | 4.45 | 5.43 | 7.95 | 5.94 | 4.11 |
| rec120 + GP-safe oil & water | 4.94 | 4.64 | 7.95 | 5.84 | 4.14 |
| **F2 rec120 + GP-safe water (FINAL)** | **4.28** | **4.55** | **8.03** | **5.62** | **3.77** |

**Key readings**
- A constant reservoir pressure fails badly in forecasting (17 %).
- Quadratic trends that look fine between tests explode when extrapolated (11 %).
- Recency weighting (τ = 120 d) improves forecasts without hurting interpolation.

## 7. PTA: synthetic benchmark with known truth

180 synthetic build-ups:
- **Events and reservoirs:** 6 real events × 30 random reservoirs.
- **Simulator:** our Laplace-space model with the real PVT, the real VFM rate history and 0.3 psi gauge noise.

Median absolute error vs the known truth:

| method | n | k_error_pct | skin_error | dP_skin_error_psi | horner_slope_error_pct |
|---|---|---|---|---|---|
| Model match (best AIC) | 180 | 1.21 | 0.09 | 1.65 | 1.21 |
| Model match (infinite) | 180 | 1.22 | 0.10 | 2.11 | 1.23 |
| Horner hourly data (F) | 180 | 2.64 | 0.25 | 6.39 | 2.71 |
| Horner P1hr measured (E) | 180 | 2.80 | 0.43 | 9.41 | 2.88 |
| **Horner auto (v5)** | **180** | **2.80** | **0.26** | **7.39** | **2.88** |
| Superposition Horner | 180 | 2.88 | 0.25 | 6.98 | 2.97 |
| Horner tp since 2025 (C) | 180 | 3.27 | 0.30 | 7.96 | 3.38 |
| Horner window 1h-end (D) | 180 | 5.11 | 0.43 | 13.14 | 5.31 |

**Choice for the deliverable:** automatic Horner. It is the required method, with 2.8 % median k error. Analytical model matching is reported alongside as a cross-check, adding boundaries and wellbore storage.

## 8. PTA: analytical model match vs Horner (real shut-ins)

| well | shut-in | type | k Horner (md) | k model (md) | skin Horner | skin model | storage C (bbl/psi) | boundary | boundary (ft) | fit RMS % | k sensitivity ±% | skin sensitivity ± |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GL-01** | **2025-04-15 08:00** | **planned build-up test** | **106.21** | **112.69** | **5.76** | **6.52** | **0.023** | **constant pressure** | **~727** | **0.32** | **3.55** | **0.43** |
| GL-02 | 2025-03-08 14:23 | unplanned shut-in | 56.23 | 58.54 | -0.51 | -0.27 | 0.016 | infinite |  | 0.08 | 2.25 | 0.12 |
| **GL-04** | **2025-06-03 08:00** | **planned build-up test** | **54.72** | **54.26** | **-1.25** | **-1.24** | **0.029** | **infinite** |  | **0.67** | **1.59** | **0.09** |
| GL-06 | 2025-07-22 03:11 | unplanned shut-in | 295.25 | 300.60 | 2.04 | 2.22 | 0.017 | infinite |  | 0.14 | 1.35 | 0.11 |
| **GL-08** | **2025-08-19 09:00** | **planned build-up test** | **42.76** | **42.86** | **-1.59** | **-1.57** | **0.010** | **infinite** |  | **0.08** | **0.10** | **0.01** |
| GL-10 | 2025-05-27 21:04 | unplanned shut-in | 179.48 | 192.10 | 1.97 | 2.54 | 0.023 | infinite |  | 0.26 | 6.41 | 0.52 |

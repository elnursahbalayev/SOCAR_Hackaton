# WellMind — Methodology

How we turned raw well data into daily rates (VFM) and reservoir properties (PTA). Every design choice is listed with the evidence behind it.

- **Experiment log** (every result table): [EXPERIMENTS.md](EXPERIMENTS.md).
- **Executed code:** [../notebooks/FINAL_VFM_PTA.ipynb](../notebooks/FINAL_VFM_PTA.ipynb).
- **Step-by-step research notebooks:** [../notebooks/research/](../notebooks/research/).

**Contents**
1. [The whole pipeline](#1-the-whole-pipeline)
2. [Data: what we found and fixed](#2-data-what-we-found-and-fixed)
3. [Physics we confirmed in the data](#3-physics-we-confirmed-in-the-data)
4. [The final VFM model](#4-the-final-vfm-model)
5. [VFM and PTA feed each other](#5-vfm-and-pta-feed-each-other)
6. [How we validated](#6-how-we-validated)
7. [How we chose the model: 47 experiments](#7-how-we-chose-the-model-47-experiments)
8. [The trap: test-separator effect](#8-the-trap-test-separator-effect)
9. [How the final model evolved](#9-how-the-final-model-evolved)
10. [Automatic PTA](#10-automatic-pta)
11. [How we validated the PTA](#11-how-we-validated-the-pta)
12. [From research to the app](#12-from-research-to-the-app)
13. [Final results](#13-final-results)
14. [Design decisions at a glance](#14-design-decisions-at-a-glance)

---

## 1. The whole pipeline

```mermaid
flowchart TB
    subgraph IN["1. Raw data (5 files)"]
        direction LR
        HF["Per-minute sensors<br/>3.9 M rows, 10 wells, 9 months"]
        CH["Choke events<br/>65 changes"]
        WT["Well tests<br/>85 labels"]
        PV["PVT + completion"]
    end
    subgraph CLEAN["2. Cleaning"]
        direction LR
        C1["Sensors: null codes, duplicates,<br/>one-minute spikes, shut-in flags"]
        C2["Labels: sign flips,<br/>gas unit error"]
    end
    subgraph FEAT["3. Features"]
        direction LR
        F1["Same minute-summary for<br/>test windows and calendar days"]
        F2["Physics features:<br/>drawdown, J, choke and tubing terms"]
    end
    subgraph MODELS["4. Models"]
        direction LR
        VFM["VIRTUAL FLOW METER<br/>oil = J x (P_res(t) - Pwf)<br/>gas = oil x Rs<br/>water = 0.5 physics + 0.5 GP"]
        PTA["AUTOMATIC PTA<br/>detect + classify shut-ins<br/>Bourdet derivative, Horner<br/>k, skin, dP skin, model match"]
    end
    OUT["5. Deliverables: VFM + PTA Excel, plots, model .pkl<br/>and the WellMind app running the same pipeline on uploads"]

    IN --> CLEAN --> FEAT --> MODELS
    VFM -- "oil rate q before each shut-in" --> PTA
    PTA -- "k and skin for J" --> VFM
    MODELS --> OUT
```

**The central choice.** There are only **85 labelled well tests for 2,730 well-days**. A model that memorises patterns cannot be trusted on new data. So the VFM is built on reservoir physics with 2–3 parameters per well, and machine learning is used only where it measurably adds information.

---

## 2. Data: what we found and fixed

```mermaid
flowchart TD
    RAW["Raw per-minute data<br/>3,917,073 rows"] --> N["Null code -999.25 to missing<br/>7,418 values"]
    N --> D["Exact duplicate rows dropped<br/>2,513 rows"]
    D --> S["One-minute spikes removed<br/>449 values (rolling median, e.g. 9,220 psia)"]
    S --> K["Choke state attached to every minute<br/>shut_in = choke 0 %"]
    K --> G["GL-06 sensor outage on 24 June<br/>day interpolated and flagged"]
    G --> CLEAN["Clean minute data"]

    LAB["Raw well tests<br/>85 rows"] --> SF["5 negative rates<br/>proved by liquid = oil + water"]
    SF --> UN["GL-06 gas in Sm3/d instead of Mscf/d<br/>all 9 tests, ratio 28.3, divided by 28.3168"]
    UN --> CLAB["Clean labels"]
```

| Decision | Why |
|---|---|
| Remove spikes with a rolling median, not by a fixed limit | A one-minute jump of thousands of psi is physically impossible in a reservoir. A median filter on the 11 neighbouring minutes removes it without touching real fast changes such as shut-ins. |
| Prove every label fix with the data itself | The sign flips are confirmed by liquid = oil + water. The gas unit error shows as gas / (oil × GOR) = 28.3, exactly m³ per Mscf. |
| **Features for tests and days come from one function** (`features.summarize_minutes`) | Training (test windows) and prediction (calendar days) must see identically built inputs. |
| Daily rate = flowing rate × fraction of the day on stream | A day with a shut-in produces less; a fully shut-in day produces 0. |

---

## 3. Physics we confirmed in the data

We built the model only after checking that the data follows these laws.

| Observation | Evidence | Consequence for the model |
|---|---|---|
| **Inflow law holds:** q<sub>oil</sub> = J·(P<sub>res</sub> − P<sub>wf</sub>) | Inverting it at each test (P<sub>res</sub> = P<sub>wf</sub> + q/J) gives a smooth, declining reservoir pressure per well | Oil = productivity index × drawdown |
| Reservoir pressure declines smoothly | The GL-01 build-up ends **1.2 psi** from the VFM reservoir-pressure trend (GL-06: 4.0 psi) | P<sub>res</sub>(t) = smooth trend per well |
| **Produced GOR = solution GOR (Rs)** | The reservoir is far above the bubble point (flowing pressure ≥ ~1,230 psi above P<sub>b</sub>). Test GOR means are within −5 … +3 % of Rs; the scatter is noise. | Gas = oil × Rs (completion file) |
| Water cut rises linearly with time | Per-well tests; quadratic trends overfit | Water cut = linear trend, recent tests weighted |
| Liquid rate ∝ choke flow term S²·√ΔP | Almost one line across all wells | Choke physics available as ML features |
| Temperature tracks the rate (r = 0.99) | ...but the physics residual vs temperature has r = −0.06 | No new information, not used |
| Remaining test-to-test scatter is noise | WC and GOR deviations correlate with no sensor and alternate in sign | We are close to the measurement-noise floor |

---

## 4. The final VFM model

```mermaid
flowchart LR
    PWF["Downhole gauge Pwf<br/>every minute"]
    KS["Own build-up k, skin<br/>(section 5)"]
    TESTS["85 cleaned well tests"]
    RS["Completion file:<br/>solution GOR Rs"]
    SAFE["Separator-safe inputs:<br/>downhole P and T, casing P,<br/>lift gas, time, J, J x Pwf"]

    KS --> J["Productivity index J<br/>Darcy: J = k h / (141.2 Bo mu (ln(re/rw) - 0.75 + s))"]
    TESTS --> PRES["Reservoir pressure P_res(t)<br/>local linear trend per well, 90-day Gaussian window,<br/>from test-implied P_res = Pwf + q/J"]
    TESTS --> WC["Water cut WC(t)<br/>linear trend, recent tests weighted (tau 120 d)"]
    J --> OIL["OIL<br/>q_o = J x (P_res(t) - Pwf)"]
    PRES --> OIL
    PWF --> OIL
    OIL --> GAS["GAS<br/>q_g = q_o x Rs / 1000"]
    RS --> GAS
    OIL --> WPH["Water, physics part<br/>q_o x WC / (100 - WC)"]
    WC --> WPH
    SAFE --> GP["Gaussian process<br/>(one-hot well id)"]
    WPH --> WATER["WATER<br/>0.5 physics + 0.5 GP"]
    GP --> WATER
    OIL --> DAY["Daily rate =<br/>flowing rate x on-stream fraction"]
    GAS --> DAY
    WATER --> DAY
```

| Component | Choice | Why (evidence in [EXPERIMENTS.md](EXPERIMENTS.md)) |
|---|---|---|
| J | Darcy from our own build-up k and skin (6 wells). Wells without a shut-in: J fitted from their tests, 25 % toward the field-typical value. | Physically grounded and self-contained. Where both exist, own-build-up J beats a J fitted from tests (oil WAPE 1.85 vs 1.93 % between tests, 4.85 vs 5.10 % forecasting). |
| P<sub>res</sub>(t) | Local linear trend (Gaussian time weights, bandwidth 90 days) | A constant P<sub>res</sub> makes the forecast error explode (17 % MAPE). Quadratic trends overfit (11 %). The local trend follows depletion that speeds up or slows down. |
| Water cut | Linear in time, recent tests weighted (τ = 120 d) | A constant WC multiplies the water error 2.2–2.5×; quadratic overfits. |
| Gas | Oil × Rs from the completion file | Above the bubble point, produced GOR = Rs. The noisy test average is worse on the clean benchmark. |
| Water ML | ½ physics + ½ Gaussian process, **separator-safe inputs only** | Lowers the forecast water error from 5.55 % to 4.92 %. Flowline-based inputs would bias normal days (section 8). |
| Oil and gas ML | None | No ML model beat physics for oil or gas, in either validation scheme. |

---

## 5. VFM and PTA feed each other

PTA needs the oil rate before each shut-in (k = 162.6·q·B·μ/(m·h)). The VFM needs J, which needs k and skin. We solve this chicken-and-egg problem by iterating (`src/own_well_properties.py`):

```mermaid
flowchart TD
    R0["Round 0: VFM with J fitted from tests"] --> Q["Oil rate q before each shut-in"]
    Q --> H["Horner analysis of every shut-in"]
    H --> KS["k and skin per well"]
    KS --> JD["Darcy J"]
    JD --> V["VFM again"]
    V --> CHK{"k changed less than 0.1 %?"}
    CHK -- "no" --> Q
    CHK -- "yes, after 3 rounds" --> FIN["Own k and skin for 6 wells<br/>(no external input)"]
```

The final model therefore uses only the provided raw data and our own analysis.

---

## 6. How we validated

```mermaid
flowchart TD
    T["85 cleaned well tests"] --> L["Leave-one-test-out<br/>85 folds: interpolation between tests"]
    T --> F["Time-forward<br/>train before 1 Jun or 1 Jul, predict next 92 days<br/>56 forecasts: what new test data asks"]
    L --> M["Official metrics per fluid<br/>R2, MAPE, WAPE (volume-weighted)"]
    F --> M
    M --> W["Paired Wilcoxon test<br/>is an improvement real or noise?"]
    W --> N["Normal-day check on ~2,700 flowing days<br/>bias vs the physics model"]
    N --> B["Organisers' benchmark<br/>used to test physics hypotheses,<br/>not to tune knobs"]
    B --> DEC["Keep a change only if it does not hurt<br/>time-forward AND passes the normal-day check"]
```

- **Why two schemes.** Leave-one-test-out measures interpolation. Time-forward measures **forecasting**, which is what the judges' later-period data requires. Several models that interpolate well forecast badly.
- **Why WAPE.** R² is ≥ 0.99 for almost every model, so it cannot rank them. WAPE (Σ|error| / Σ|true|) weights high-rate wells, like the official score.
- **Why the normal-day check.** Every label is a test, taken under test-separator conditions. Normal days can only be checked indirectly (section 8).

---

## 7. How we chose the model: 47 experiments

```mermaid
flowchart TD
    ALL["47 experiments, same folds, same metrics"] --> A["A. Pure ML on raw or physics features<br/>Ridge, kNN, SVR, RF, ExtraTrees,<br/>LightGBM, XGBoost, CatBoost, MLP, GP"]
    ALL --> B["B. Physics per well<br/>+ ablations of each component"]
    ALL --> C["C. Hybrids<br/>residual learning, physics as feature,<br/>engineered targets liquid + WC + GOR"]
    ALL --> D["D. Sim-to-real<br/>pre-train ML on a calibrated simulator,<br/>fine-tune on real tests"]
    ALL --> E["E/F. Ensembles<br/>physics + GP blends"]

    A --> A1["Trees and MLP overfit 85 labels:<br/>forecast WAPE 7.9 - 17.9 %<br/>none better than repeating the last test (7.9 %)"]
    A --> A2["Smooth models best: GP 5.2 %<br/>but +10 % oil on normal days"]
    B --> B1["Physics 4.9 %, unbiased"]
    C --> C1["Residual ML: correction goes to 0, nothing left to learn<br/>Best C3 Ridge 4.41 % but water -16.7 % on normal days: REJECTED"]
    D --> D1["Helps trees a lot (LightGBM 8.1 to 5.1 MAPE)<br/>still below physics"]
    E --> E1["Physics oil and gas + GP water,<br/>separator-safe: 4.49 %, unbiased: SELECTED"]
```

| Lesson | Evidence |
|---|---|
| With 85 labels, flexible models memorise | XGBoost forecast WAPE 13.5–17.9 %, LightGBM 9.5–11.7 %, MLP 15.4 % vs physics 4.9 % |
| Smooth beats complex | GP and Ridge beat every tree model |
| Physics leaves almost nothing systematic | Residual Ridge on top of physics learns a correction of ≈ 0 |
| Synthetic pre-training is real but not enough | It improves trees by 23–43 % (e.g. LightGBM MAPE 8.1 → 5.1 %); still below physics |
| **A model can win the table and still be wrong** | C3 Ridge had the lowest forecast WAPE (4.41 %) but under-predicts water by 16.7 % on normal days → rejected |

---

## 8. The trap: test-separator effect

During a well test the flow goes to the **test separator**, which runs at lower pressure than the production line:

```mermaid
flowchart LR
    subgraph NORMAL["Normal day"]
        W1["Well"] --> PL["Production line<br/>higher pressure"]
    end
    subgraph TEST["Well-test day"]
        W2["Well"] --> TS["Test separator<br/>17-29 psi lower flowline pressure"]
        TS --> MORE["slightly lower Pwf,<br/>slightly higher rate"]
    end
    MORE --> LABEL["All 85 labels come from this regime"]
    LABEL --> RISK["ML using flowline / choke pressures<br/>learns a test-only relation"]
    RISK --> BIAS["On normal days: XGBoost +48.6 %,<br/>LightGBM +45.9 %, GP +9.4 % oil"]
    LABEL --> OURS["Physics uses Pwf (measured every day)<br/>GP uses separator-safe inputs only"]
    OURS --> OK["Our model: +0.8 %"]
```

- **Why validation misses it.** Leave-one-test-out and time-forward cannot see this, because every validation point is a test. We found it by comparing models on **normal flowing days** against the physics model.
- **Why physics is a fair reference here.** Its reservoir pressure is confirmed independently by the build-up (1.2 psi).
- **The fix.** The water GP uses only inputs the separator does not disturb.

---

## 9. How the final model evolved

```mermaid
flowchart TB
    S0["Physics per well<br/>(B)"] --> S1["+ Gaussian process for water<br/>(E2)"]
    S1 --> S2["+ recency weighting, tau 120 d<br/>forecast-driven"]
    S2 --> S3["+ separator-safe GP inputs<br/>normal-day bias 10 % to 1 %"]
    S3 --> S4["+ gas = oil x Rs<br/>benchmark 20.975 to 21.183"]
    S4 --> S5["+ local P_res trend (90 d)<br/>forecast oil WAPE 3.63 to 3.48, p = 0.002"]
    S5 --> S6["J from our own build-up analysis<br/>FINAL (v5): self-contained, VFM 21.089 / 30"]
```

**Hypotheses tested against the organisers' benchmark** (each kept only if time-forward validation agreed):

| Hypothesis | Result | Decision |
|---|---|---|
| Gas = oil × Rs instead of the noisy test-average GOR | 20.975 → **21.183** | kept |
| Water cut from BSW instead of metered water cut | 21.118 (worse) | rejected |
| Local reservoir-pressure trend | **21.224**; forecast oil WAPE 3.63 → 3.48 (p = 0.002) | kept |
| Local water-cut trend | not significant (p 0.28 / 0.75) | rejected |
| Temperature as extra information | residual vs temperature r = −0.06 | rejected |
| Build-up pressures as reservoir-pressure anchors | trend already matches (1.2–4 psi) | no gain |

Step-by-step contribution of each component in the final configuration ([EXPERIMENTS.md](EXPERIMENTS.md#2-final-model-step-by-step)):

| Step | Oil WAPE between / forecast | Water | Gas |
|---|---|---|---|
| Repeat last test (naive) | 3.14 / 8.16 | 4.03 / 5.32 | 5.61 / 10.22 |
| Physics: J × drawdown, trends | 2.08 / 4.12 | 3.39 / 5.53 | 4.33 / 5.39 |
| + recent tests weighted | 2.16 / 3.69 | 3.50 / 5.71 | 4.32 / 4.95 |
| + gas = oil × Rs | 2.16 / 3.69 | 3.50 / 5.71 | 4.40 / 6.00* |
| + local reservoir-pressure trend | 2.03 / 3.52 | 3.38 / 5.55 | 4.33 / 5.63 |
| **+ GP for water = final** | **2.03 / 3.52** | **3.08 / 4.92** | **4.33 / 5.63** |

\* Against the *test labels*, Rs looks slightly worse for gas. The labels carry separator measurement noise, while Rs is the clean physical constant; the benchmark, scored against the noise-free truth, improved.

---

## 10. Automatic PTA

```mermaid
flowchart TB
    A["Minute downhole pressure<br/>+ choke events"] --> B["Detect shut-ins<br/>choke = 0 %"]
    B --> C{"Classify"}
    C -- "planned build-up test" --> D["Log-binned build-up<br/>20 points per log cycle"]
    C -- "unplanned shut-in" --> X["Analysed and shown,<br/>not reported"]
    D --> E["Bourdet derivative vs<br/>Agarwal equivalent time (L = 0.15)"]
    E --> F["Radial-flow window =<br/>flattest derivative after the storage hump<br/>(at least 0.5 log cycle)"]
    F --> G["Horner least squares on minute data<br/>Pws = P* - m log((tp + dt)/dt)"]
    G --> H["k = 162.6 q B mu / (m h)<br/>s = 1.151 [(P1hr - Pwf)/m - log(k/(phi mu ct rw2)) + 3.23]<br/>dP skin = 0.869 m s"]
    H --> I["Analytical model match<br/>storage + skin + boundary, Stehfest, AIC"]
    I --> J["Sensitivity, interpretation,<br/>plots, Excel"]
```

**Classification** (textbook definition of a build-up *test*: a planned operation; Earlougher, Horne):

```mermaid
flowchart TD
    S["Shut-in detected"] --> L{"Duration at least 24 h?"}
    L -- "no" --> U["Unplanned shut-in<br/>GL-02 5.7 h, GL-06 8.0 h, GL-10 9.0 h"]
    L -- "yes" --> R{"Choke data at reopening?"}
    R -- "no" --> P["Planned build-up test<br/>(duration decides)"]
    R -- "yes" --> ST{"Staged ramp-up,<br/>at least 2 choke steps?"}
    ST -- "yes" --> P2["Planned build-up test<br/>GL-01 72 h, GL-04 48 h, GL-08 72 h<br/>all reopened in 3 steps, started on the hour"]
    ST -- "no" --> U
```

| Input | Choice | Why |
|---|---|---|
| Rate q | Mean VFM oil rate of the 7 days before the shut-in | A single day is noisy; k scales 1:1 with q |
| P<sub>wf</sub>(Δt=0) | Mean downhole pressure of the 24 h before the shut-in | Robust to minute noise |
| Producing time | t<sub>p</sub> = 24·N<sub>p</sub>/q, N<sub>p</sub> = cumulative oil at 1 Jan 2025 + VFM oil since | Horner definition; t<sub>p</sub> ≫ Δt, so its effect is < 0.5 % |
| Window | Automatic from the derivative; adjustable by hand in the app | The standard way to find radial flow without a human picking the line |
| Boundaries | Accepted only if AIC improves by > 4 and the boundary lies inside the radius of investigation | Avoids over-interpreting the end of the build-up |

---

## 11. How we validated the PTA

There is no "true k" for real data. So we built a **synthetic benchmark with known answers** (`src/pta_synthetic.py`, `src/run_pta_benchmark.py`):

```mermaid
flowchart LR
    R["Random reservoir:<br/>k, skin, storage, boundary"] --> SIM["Laplace-space simulator<br/>+ REAL PVT<br/>+ REAL VFM rate history<br/>+ 0.3 psi gauge noise"]
    SIM --> BU["180 synthetic build-ups<br/>6 real events x 30 reservoirs"]
    BU --> METH["8 interpretation methods"]
    METH --> ERR["Error vs the known truth"]
```

| Method | Median \|k error\| | Median \|skin error\| |
|---|---:|---:|
| Model match (best AIC) | 1.2 % | 0.09 |
| Horner on hourly data | 2.6 % | 0.25 |
| **Horner, automatic (ours, deliverable)** | **2.8 %** | **0.26** |
| Superposition Horner | 2.9 % | 0.25 |
| Horner, t<sub>p</sub> since 2025 | 3.3 % | 0.30 |
| Horner, fixed window 1 h → end | 5.1 % | 0.43 |

**Checks on the real data:**
- **Two independent methods agree:** the GL-01 build-up ends **1.2 psi** from the VFM reservoir pressure, which was inferred from well tests only.
- **Sensitivity of the automatic choices** (window, smoothing, minimum width):
  - k varies ±1.6 % (GL-04) and ±0.1 % (GL-08);
  - GL-01 varies ±3.6 % in k and ±0.43 in skin, because its pressure support cuts radial flow short.
- **Model match vs Horner:** GL-01 112.7 vs 106.2 md, GL-04 54.3 vs 54.7 md, GL-08 42.9 vs 42.8 md.

---

## 12. From research to the app

```mermaid
flowchart TB
    U["Engineer uploads raw CSVs<br/>(any subset, any file names)"] --> V["validate<br/>recognise files by columns"]
    V --> C["clean"]
    C --> FE["features"]
    FE --> M["VFM with the stored model .pkl<br/>(no retraining)"]
    M --> P["PTA: detect, classify, Horner,<br/>model match"]
    P --> X["Excel + report + plots"]
    H[("Well history<br/>cumulative oil, last choke state,<br/>PVT, completion")] --> M
    H --> P
    X --> UI["Pages: Data, VFM, Well Test Analysis,<br/>Insights, Report"]
```

- **Same code.** The app runs exactly the code in `src/` that produced the deliverables (`src/pipeline.py`).
- **Proof.** Uploading only two months of raw files reproduces the GL-08 build-up result to the last digit.
- **Missing files fall back to the stored field data:**
  - PVT and completion → the stored ones;
  - choke file → shut-ins detected from gas lift stopping;
  - a later period → cumulative oil and choke state carried over.

---

## 13. Final results

**VFM**: official metrics, 85 tests

| Fluid | WAPE between tests | WAPE forecast 0–3 months | R² between / forecast |
|---|---:|---:|---|
| Oil | 2.03 % | 3.52 % | 0.999 / 0.996 |
| Water | 3.08 % | 4.92 % | 0.997 / 0.988 |
| Gas | 4.33 % | 5.63 % | 0.995 / 0.993 |

**PTA**: planned build-up tests

| Well | k (md) | Skin | ΔP skin (psi) | Horner m (psi/cycle) | Flow efficiency | Boundary |
|---|---:|---:|---:|---:|---:|---|
| GL-01 | 106.21 | +5.76 | 115.64 | 23.12 | 0.68 | constant pressure, ~730 ft |
| GL-04 | 54.72 | −1.25 | −158.29 | 145.34 | 1.12 | infinite-acting |
| GL-08 | 42.76 | −1.59 | −127.99 | 92.77 | 1.16 | infinite-acting |

**Organisers' benchmark:** VFM 21.09 / 30, PTA 12.70 / 20.

---

## 14. Design decisions at a glance

| # | Decision | Alternatives tested | Deciding evidence |
|---|---|---|---|
| 1 | Physics-first VFM | 10 pure-ML models, hybrids, sim-to-real | Lowest forecast error, unbiased on normal days |
| 2 | Darcy J from **own** build-ups | J fitted from tests | Oil WAPE 4.85 vs 5.10 % forecasting where both exist; physically grounded; self-contained |
| 3 | Local linear P<sub>res</sub> trend | constant, linear, quadratic, slope shrinkage | Constant 17 % / quadratic 11 % forecast MAPE; local trend p = 0.002 better |
| 4 | Gas = oil × Rs | test mean / median / linear GOR | Physics above P<sub>b</sub>; benchmark +0.21 |
| 5 | Water = ½ physics + ½ GP | GP only, physics only, other blends | Forecast water 5.55 → 4.92 % |
| 6 | GP on separator-safe inputs | GP on all physics features | Normal-day oil bias +9.4 % → +0.8 % |
| 7 | Reject the "best-scoring" C3 Ridge | – | −16.7 % water on normal days |
| 8 | Report only planned build-up tests | report all 6 shut-ins | Textbook definition: ≥ 24 h + staged restart |
| 9 | Automatic Horner for the deliverable; model match as cross-check | 7 other methods | 2.8 % k error on 180 synthetic tests; the required method |
| 10 | q = 7-day mean VFM rate; P<sub>wf</sub> = 24-h mean | rate on the shut-in day, last minute | Robust; k scales 1:1 with q |

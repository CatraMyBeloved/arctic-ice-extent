# Results Log

The running record of what each experiment **found** — results, interpretation, honest
negatives, and the bugs we caught along the way. The forward-looking plan lives in
[`project_plan.md`](project_plan.md); detailed specs for not-yet-run experiments live in
[`experiment_designs.md`](experiment_designs.md).

**Source of truth for numbers**: `results/model_comparison.csv`, written via
`log_model_results()` and read back by `notebooks/07_model_comparison.ipynb`, which ranks all
models and regenerates predictions from any checkpoint bundle in `models/`. This document adds
narrative and interpretation on top; if a number here ever disagrees with the CSV, the CSV wins.

All comparisons are only valid within the same `scale` tag (`daily`, `monthly`, `multistep_7d`, ...).

---

## The story so far

1. **Daily pan-Arctic extent is brutally autocorrelated.** Persistence (`y_t+1 = y_t`) achieves
   RMSE ≈ 0.087 Mkm² on the 2020–2023 test era — that, not climatology (≈ 1.0 Mkm²), is the bar
   any daily model has to beat.
2. **A plain univariate LSTM beats that bar, modestly but significantly.** RMSE 0.073 Mkm²
   (skill +0.168) on the fixed split; skill +0.296 ± 0.088 across a 5-fold expanding-window
   backtest, beating persistence in 5/5 folds.
3. **Then we hit the floor.** The best daily model's RMSE (0.0593 Mkm²) sits almost exactly on
   NSIDC's own documented daily-extent measurement uncertainty (~0.06 Mkm², 2σ). Further
   1-day-horizon tuning is chasing the precision limit of the data, not model error.
4. **Uncertainty quantification produced two instructive failures.** A 10-seed ensemble is badly
   overconfident (90% PICP 0.17) because all seeds converge to nearly the same function; MC
   Dropout covers well (PICP 0.94) only by being ~36× wider. Neither method's uncertainty
   correlates with actual error — the dominant error source is *aleatoric* (genuine day-to-day
   unpredictability), which weight-space methods can't express.
5. **So the open questions moved.** Extended horizons (where does skill decay to zero?) and
   distribution modelling (quantile / NLL / VAE — honest intervals for aleatoric noise) are where
   the remaining headroom is. See the roadmap in `project_plan.md`.

---

## Results by experiment

### 03a — SARIMA baselines (monthly scale)

* Two SARIMA models on monthly-aggregated extent: SARIMA(1,0,1)×(0,1,1,12) on raw values,
  SARIMA(2,0,2)×(1,0,1,12) on anomalies. Train 1989–2019 / test 2020–2023 (48 months),
  1-month-ahead walk-forward.
* **Result**: SARIMA_raw RMSE ≈ 0.227 Mkm², skill +0.88 vs (monthly) persistence, +0.72 vs
  climatology. Logged at `scale="monthly"` — not comparable to the daily numbers below.

### 03b — Persistence & climatology baselines (daily scale)

* Evaluated on the 2020–2023 daily pan-Arctic test set via `src/evaluation_utils.py`.
* **Result**: persistence RMSE ≈ 0.087 Mkm² (the hard bar at daily scale); climatology ≈ 1.0 Mkm².
  Seasonal breakdown and figure produced.

### 04 — Univariate LSTM (daily)

* 2-layer LSTM (64 hidden, dropout 0.2), extent only, three-way split
  (train 1989–2014 / val 2015–2019 / test 2020–2023), seeded, via the shared
  `src/lstm_utils.py` engine.
* **Result**: RMSE 0.073 Mkm², beats persistence (skill +0.168). Logged as
  `LSTM_Basic_Univariate`. Winter-vs-summer seasonal breakdown done.

### 05 / 06 — Multivariate & seq2seq LSTM

* Refactored onto the shared engine, **pending a GPU-box run** (they need the ERA5 parquet
  store). Their rows in `model_comparison.csv` fill in automatically once run; 07 picks them up.

### 07 — Model comparison (self-updating)

* Ranked comparison per scale, Diebold-Mariano significance vs persistence & climatology,
  Holm-Bonferroni correction across the family of tests.
* **Result (laptop, univariate family)**: every univariate LSTM variant beats persistence
  significantly *except MC Dropout* (DM ≈ −9.4 to −16.9, p ≈ 0, survives Holm-Bonferroni). Best
  daily model: `09_ensemble_seed8`, RMSE 0.0593 Mkm². 2020 flagged as the highest-error test
  year — independently confirmed by 08's backtest.

### 08 — Expanding-window backtesting (univariate)

* 5 folds (test years 2019–2023), each with its own 2-year validation window and all prior
  history as training data; the univariate architecture retrained per fold.
* **Result**: LSTM beats persistence in **5/5 folds** — RMSE 0.0623 ± 0.0102 vs persistence's
  0.0882 ± 0.0036, skill +0.296 ± 0.088. Higher skill than the single fixed-split result
  (+0.168), likely from more training data per fold. 2020 is consistently the weakest fold.

### 09 — 10-seed LSTM ensemble (univariate)

* 10 independent trainings differing only in random init; prediction intervals from ensemble
  spread.
* **Result**: member RMSE 0.0602 ± 0.0004 — ten seeds converge to nearly the same function.
  Ensemble mean RMSE 0.0597, skill +0.316 vs persistence (DM p ≈ 0). The ensemble beats a
  *typical* member (DM p = 0.022) but not the single best member.
* **Negative result, the important one**: 90% prediction-interval PICP is only **0.170** — badly
  overconfident. Inter-seed (epistemic) spread massively underestimates real error because there
  is almost no inter-seed spread to begin with.
* Cost: ~223 s/member on CPU, 2,227 s for all 10.

### 10 — MC Dropout (univariate)

* Dropout enabled at inference, 50–100 forward passes; needed a new `head_dropout` param on
  `IceExtentLSTM` since between-layer LSTM dropout alone doesn't reach the output layer. Reuses
  09's checkpoints, no retraining.
* **Result**: with `head_dropout=0.3` the point forecast got *worse* — RMSE 0.095, skill
  **−0.089** vs persistence (not significant, DM p = 0.136); too much regularization for this
  small a hidden layer. 90% PICP **0.938** (close to nominal, unlike the ensemble) but MPIW
  **~36× wider** than the ensemble's — well-covered mostly by being very wide, not by being
  precisely calibrated.

### Residual diagnostics (after 09/10)

* Both UQ methods' uncertainty correlates weakly with actual |error| (ensemble 0.04, MC Dropout
  0.15) → the model's error is dominated by **aleatoric**, not epistemic, uncertainty.
* Ljung-Box shows residual autocorrelation; `corr(|residual|, |actual day-to-day change|) =
  0.456` — error concentrates on high-volatility (rapid melt/freeze) days.
* Best daily RMSE (0.0593 Mkm²) ≈ NSIDC's documented measurement uncertainty (~0.06 Mkm², 2σ):
  the 1-day horizon is at or near the data's own noise floor.
* These three findings jointly motivate the extended-horizon, distribution-loss, and VAE designs
  in `experiment_designs.md`.

---

## Bugs caught and fixed

Documented deliberately — catching these is a core learning outcome of the project.

1. **Test-set leakage in the original LSTM notebooks.** Early stopping and checkpoint selection
   watched *test* loss (2020–2023), so the test set leaked into model selection and the old
   results were optimistically biased. Fixed by rebuilding everything on a **three-way temporal
   split** (train 1989–2014 / validation 2015–2019 / test 2020–2023) in `src/lstm_utils.py`:
   early stopping, checkpointing, and LR scheduling are driven by validation loss only, and the
   test era is touched once, for final evaluation. All logged results postdate this fix.
2. **Silent validation-year reuse in 07's checkpoint auto-discovery.** 07 originally
   re-evaluated 08's per-fold checkpoint bundles on the shared 2020–2023 window — silently
   reusing each fold's own validation years as test data. Fixed by excluding
   `08_backtest_fold_*` bundles from 07's shared-test-era comparison by name.

---

## Key lessons so far

* **Baselines first, always.** Persistence looks trivial and eats naive models alive at daily
  scale; a model that "predicts sea ice with RMSE 0.07" means nothing without that reference.
* **Model selection must never see the test set** — and the failure mode is subtle (early
  stopping is model selection too).
* **Know the measurement uncertainty of your target.** We spent effort approaching a floor set
  by the sensor, not the model. Checking NSIDC's documented uncertainty earlier would have
  redirected effort sooner.
* **Seeds converging ≠ predictions trustworthy.** Ensemble spread only measures epistemic
  uncertainty; when the loss landscape funnels all seeds to the same function, the spread says
  nothing about real error.
* **MSE smooths.** On a near-persistence series, the conditional mean under-shoots and lags
  rapid transitions — exactly where our error concentrates. Whether any loss change can fix
  that (vs. it being irreducible noise) is an open, explicitly gated question
  (see `experiment_designs.md`, objective-function experiments).

# Experiment Designs

Detailed specs for planned (not-yet-run) work. The prioritized order lives in
[`project_plan.md`](project_plan.md); once an experiment completes, its findings move to
[`results_log.md`](results_log.md) and its entry here shrinks to a pointer.

---

## Core path

### E1 — Simple ML baselines

**Question**: does the LSTM beat anything smarter than persistence? Right now the comparison
table has trivial baselines and neural nets, with nothing in between — on tabular lagged
features at daily resolution, well-tuned Ridge or gradient boosting is often competitive with
an LSTM, and if it is here, that's a headline finding.

* **Prerequisites**: none (extent-only feature set runs on the laptop; ERA5 features optional).
* **Scope**:
  * Linear regression, Ridge, Lasso, Random Forest, XGBoost on lagged extent features
    (t−1, t−7, t−14, t−30) + day-of-year sin/cos, 1-day-ahead, daily scale.
  * Same three-way temporal split as the LSTMs (train 1989–2014 / val 2015–2019 for
    hyperparameters / test 2020–2023); results logged at `scale="daily"`.
  * Multi-horizon predictions (+7, +14, +30 days) as direct variants, with skill scores vs
    persistence and climatology — these also give the E3 skill-decay curve a non-neural
    reference line.
* **Done when**: all five models appear in 07's daily table with DM significance vs persistence,
  with multi-horizon rows at +7/+14/+30.
* Notebook: `03c_ml_baselines.ipynb` (baseline-stage numbering; the old plan's `05_ml_baselines`
  name collided with `05_multivariate_lstm`).

### E2 — Multivariate & seq2seq runs (GPU box)

**Question**: do ERA5 climate features add skill over the univariate model — at 1 day (is the
~0.06 Mkm² floor data-precision-limited or information-limited?) and at 7 days?

* **Prerequisites**: ERA5 parquet store present (GPU box).
* **Scope**:
  * Run the already-refactored `05_multivariate_lstm.ipynb` (variants: `climate`,
    `climate+cyclical`, `climate+lags`) and `06_seq2seq_lstm.ipynb` (univariate / climate /
    climate+cyclical at horizon 7).
  * Rerun `07_model_comparison.ipynb` — the 05/06 rows fill in automatically.
  * Rerun the 10-seed ensemble (09) and MC Dropout (10) on 05's best variant, so the
    uncertainty-quantification comparison is grounded in the best architecture rather than the
    univariate stand-in.
* **Out of scope**: new architecture variants; hyperparameter sweeps.
* **Done when**: 05/06/ensemble rows are in `model_comparison.csv` with significance tests, and
  the 1-day answer (climate features move the floor or they don't) is written into
  `results_log.md`.

### E3 — Extended horizons & the skill-decay curve

**Question**: how far into the future does the model beat trivial baselines at all? This is the
project's headline figure. Motivation: the 1-day RMSE sits on NSIDC's measurement noise floor
(see `results_log.md`), so lead time — not 1-day accuracy — is where the open questions are.

* **Prerequisites**: none strictly (univariate inputs suffice); E2 first if climate features
  prove useful at 7 days.
* **Scope**:
  * Two dedicated direct models: `forecast_horizon=14` and `forecast_horizon=30`, each its own
    `IceExtentLSTM` + `train_model` run on the shared engine (7-day already exists from 06).
  * **Direct, not MIMO**: one model per target horizon, not one wide-output model sliced
    afterward. A single model trained with one aggregate loss across a 30-day output must
    compromise across horizons of very different difficulty (day-1 easy, day-30 hard) — the
    classic direct-vs-MIMO tradeoff (Taieb & Hyndman).
  * **No autoregressive rollout**: feeding predictions back in only works cleanly univariate;
    for climate-feature models it silently requires *future* weather at every step — data that
    doesn't exist for a genuine forecast (ERA5 is reanalysis, not a forecast product). Published
    sea-ice forecasters converge on direct output from historical-only inputs: IceNet (Andersson
    et al. 2021, *Nat. Commun.*) predicts all 6 future months in one forward pass; ECMWF's SEAS5
    solves it dynamically instead. We do the same.
  * The skill-decay curve: RMSE + skill vs persistence & climatology across 1/7/14/30-day lead
    times (stitching in 04's 1-day and 06's 7-day results), the zero-skill crossover lead time,
    and error/uncertainty growth vs lead time.
* **Out of scope**: the MIMO-vs-direct ablation (a good later question, but only *after* the
  curve exists); loss-function variants (that's E4); attention (E5); horizons beyond 30 days.
* **Done when**: the skill-decay figure exists with all four lead times and a
  zero-skill-crossover estimate, and 14/30-day rows are logged at their own `scale` tags.
* Notebook: `11_extended_horizon_seq2seq.ipynb`.

### E4 — Objective-function experiments (gated)

**Question**: is the leftover error structure (concentrated on rapid melt/freeze transitions,
`corr(|residual|, |Δy|) = 0.456`) extractable signal, or an irreducible noise floor — and if
extractable, does changing the training objective recover it?

**Caveat up front, so we don't fool ourselves**: MSE's smoothing is the *optimal* hedge under
irreducible uncertainty. If transition timing isn't predictable from the inputs, no loss change
helps — it only trades "smooth and calibrated" for "sharp and mistimed", which is usually worse
on any honest metric. Hence the gate.

* **Prerequisites**: E3's dedicated per-horizon models (these experiments layer onto them —
  MSE's smoothing penalty grows with lead time, so extended horizons are where an
  anti-smoothing objective has the most to prove). ERA5 features (E2) for the gate regression.
* **Scope**:
  * **Step 0 — extractability gate (mandatory, first)**: regress the univariate model's
    residuals on predictors it didn't use (ERA5 variables, day-of-year regime, local
    volatility). If they significantly cut residual autocorrelation (Ljung-Box) or explain
    residual variance → the structure is missed *signal*; proceed to A and B. If the
    autocorrelation survives every available feature → it's plausibly colored measurement
    noise; **skip A/B entirely** and do only C.
  * **A — differenced target**: predict `Δy_t` instead of the level (a `target_mode` switch on
    `SequenceDataset`), reconstruct the level by cumulative sum, always evaluate on the
    reconstructed level. Watch multi-step error accumulation.
  * **B — rate-of-change losses**: gradient-difference term `MSE(ŷ,y) + λ·MSE(Δŷ,Δy)`;
    volatility-weighted MSE `Σ wₜ(ŷₜ−yₜ)²` with `wₜ ∝ |Δyₜ|`.
  * **C — distribution losses**: quantile/pinball (several quantiles → robust median point
    forecast + state-dependent intervals that widen during transitions) and/or Gaussian NLL
    (μ, σ² → honest heteroscedastic aleatoric uncertainty). These attack the overconfident
    intervals *and* the transition problem in one move, are philosophically the same family as
    the E6 VAE's ELBO, and are likely the highest-value objective change given the
    aleatoric-error finding. Reuse the `picp`/`mpiw`/reliability harness from 09/10.
  * Evaluation: train on the new objective, still report RMSE in 07 for comparability, and add
    **one** shape/timing metric (RMSE of Δ, or melt-onset timing error) so a sharper forecast is
    visible rather than silently penalized. Every variant vs the plain-MSE model on identical
    splits.
* **Stretch within this experiment** (multi-horizon only, follow-on if A/B show promise):
  **DILATE / soft-DTW** — a shape+timing loss built specifically to stop smoothed, time-lagged
  multi-step forecasts; most relevant to the 14/30-day models from E3, heavier to implement.
* **Out of scope**: combining objectives (e.g. differenced + weighted); re-tuning architecture
  per objective.
* **Done when**: the gate verdict is written into `results_log.md`, the gated subset of A/B/C is
  trained and in 07's table with the shape metric, and the "can a loss fix the smoothing?"
  question has a documented yes/no.

### E-final — Write-up

Not an experiment, but a core-path deliverable with the same right to a done-when line: distill
the narrative (persistence bar → LSTM wins → noise floor → aleatoric error → skill-decay curve)
into the README plus `results_log.md`'s story section, with the skill-decay figure as the
headline. **Done when** a reader who opens only the README understands what was found, what
failed, and why the failures were informative.

---

## Stretch (only after the core path is done)

### E5 — Enhanced encoder-decoder (attention seq2seq)

**Question**: do standard seq2seq enhancements — attention, teacher forcing, a bidirectional
encoder — improve on the vanilla encoder-decoder from 06/E3, and what do the attention weights
reveal about the temporal dependencies the model uses?

* **Scope**:
  * Implement an attention mechanism for the Seq2Seq architecture.
  * Add teacher forcing during training (scheduled sampling).
  * Test a bidirectional encoder for improved context modeling.
  * Multi-horizon outputs: 7-day, 14-day, 30-day forecast sequences.
  * Compare the enhanced Seq2Seq vs the vanilla versions (06 and E3's models) on identical
    splits.
  * Analyze attention weights to understand temporal dependencies.
* **Done when**: enhanced-vs-vanilla comparison rows exist at each horizon and the
  attention-weight analysis is written up.
* Notebook: `12_attention_seq2seq.ipynb`.

### E6 — Predictive VAE (variational seq2seq)

**Question**: does explicitly modelling a *distribution* of futures produce intervals that are
both better calibrated and more error-correlated than the ensemble (PICP 0.17) and MC Dropout
(PICP 0.94 but ~36× wide, corr ≈ 0.15)? Motivated directly by the aleatoric-error finding: a
VAE's latent variable models plausible-future spread, not weight uncertainty.

* **Prerequisites**: E3's 14-day setup; the two infra items below.
* **Infra (small, scoped)**:
  * Fix `load_checkpoint` in `lstm_utils.py` — it hardcodes rebuilding `IceExtentLSTM`
    regardless of the saved `model_class`; needs a small registry dispatch before a second model
    class can round-trip.
  * New `src/vae_utils.py` mirroring `lstm_utils.py` (`PredictiveVAE`, ELBO loss, KL annealing
    config, `train_vae`, `sample_futures`), reusing `SequenceDataset`, `set_seed`,
    `get_device`, checkpointing, and the evaluation utilities as-is.
* **Architecture**: LSTM encoder over the 30-day window (extent + day-of-year sin/cos) → heads
  `mu`, `logvar` (latent_dim 8–16, `hidden_size` 64) → reparameterize → LSTM decoder initialized
  from `z`, stepped autoregressively with only future day-of-year sin/cos as per-step input —
  autoregression is fine *here* because calendar arithmetic is always knowable in advance (the
  E3 objection was to feeding forward unknowable future weather, not to autoregression itself).
  Loss: MSE reconstruction + β·KL (closed-form vs unit Gaussian).
* **Training details that are not optional**: KL annealing (ramp β from 0 over the first
  ~30–50% of training) to avoid posterior collapse; watch `kl_loss` — if it pins to ~0,
  mitigations in order: lower β, free-bits, feed `z` into every decoder step. Sweep only
  `latent_dim` and β; keep `hidden_size`/`num_layers` at 06/11's values.
* **Evaluation**: point forecast from `mu` in 07's table like any model; N=50 samples of `z`
  (matching MC Dropout's pass count) → PICP/MPIW @ 90%, three-way reliability diagram vs 09/10,
  `corr(uncertainty, |error|)`.
* **Out of scope**: 30-day until 14-day is stable (no collapse, sane calibration); Gaussian-NLL
  per-step variance head (stretch-of-a-stretch); conditioning on ERA5.
* **Done when**: the three-way calibration comparison exists and answers whether sampled-`z`
  intervals beat both prior methods.
* Notebook: `13_predictive_vae.ipynb`.

### E7 — Advanced features (regional models, more ERA5 variables)

**Question**: does modelling the 14 Arctic regions individually — with richer atmospheric
forcing — beat the pan-Arctic aggregate approach?

* **Scope** (longer-term, from the original Phase 5 / M6):
  * Expand to regional models for all 14 Arctic regions (Tier B).
  * Add more ERA5 variables (winds, geopotential, longwave radiation).
  * Cross-regional feature engineering.
  * Implement the ice-edge band approach (Tier C).
  * Advanced spatio-temporal modeling.
  * Expanded evaluation across regions.
* **Done when**: regional models are evaluated across all regions on the same protocol and the
  regional-vs-pan-Arctic comparison is documented.
* Not yet designed in detail — spec this out (per-region baselines, how regional skill
  aggregates, which regions first) when it moves onto the active roadmap.

### E8 — Spatial-temporal CNN-LSTM (Phase 6)

**Question**: do gridded weather fields carry predictive signal that regional aggregates throw
away? The big learning payoff (zarr, regridding, CNN encoders, interpretability) and the real
GPU workload — strictly after the core path.

* **Stage 1 — spatial ERA5 preprocessing** (`14_era5_spatial_preprocessing.ipynb`):
  * Download gridded ERA5 data (not regional aggregates): T2M, MSLP, SST, sea ice concentration.
  * Regrid to Arctic Stereographic projection (EPSG:3411, Arctic-focused equal-area grid).
  * Downsample to computationally feasible resolution (64×64 and 128×128 grids).
  * Create multi-channel "image" time series: each timestep = [T2M, MSLP, SST, SIC] stack,
    per-variable standardization.
  * Store as zarr arrays or HDF5 for efficient sequential access.
* **Stage 2 — model implementation** (`15_cnn_encoder.ipynb`, `16_cnn_lstm_hybrid.ipynb`):
  * CNN spatial encoder — input (batch, time_steps, channels, height, width); architecture
    options: ResNet-style (3-4 conv blocks with residual connections) or small VGG-style
    (3-5 conv layers with pooling, 256-512 feature vector). Optionally pre-train the CNN on an
    auxiliary task (e.g. predict SIC from weather fields).
  * LSTM temporal decoder: 2-3 layers over the encoded sequence → scalar pan-Arctic extent.
    Multi-horizon variants: single-step (t+1) and Seq2Seq (t+1…t+7, extendable to 14-30 days as
    direct multi-step output — no autoregressive rollout, same reasoning as E3).
  * Hybrid architecture: concatenate CNN-encoded spatial features with tabular features
    (climatology, lagged extent, day-of-year) before the LSTM.
  * Training strategy: modular pre-training (optional) then end-to-end joint optimization;
    gradient checkpointing for the CNN and sliding-window batches for memory efficiency; data
    augmentation (random crops, spatial shifts, where applicable to the Arctic grid); spatial
    dropout in the CNN, temporal dropout in the LSTM.
* **Stage 3 — evaluation & analysis** (`17_cnn_lstm_evaluation.ipynb`):
  * Performance comparison vs the best aggregated-feature LSTM on identical splits.
  * Ablation studies: CNN features only vs tabular only vs hybrid; spatial resolution impact
    (64×64 vs 128×128); number of input channels (T2M only vs full 4-channel stack).
  * Spatial interpretability: CNN activation maps (GradCAM or saliency), which Arctic regions
    drive predictions, seasonal patterns in feature importance.
  * Computational cost analysis: training time vs aggregated models (likely 5-10× slower),
    inference latency, memory requirements and scalability.
* **Done when**: the deliverables exist — preprocessing pipeline, trained CNN-only / LSTM-only /
  hybrid models, the evaluation notebook with ablations and interpretability figures, and
  documentation of the computational trade-offs.

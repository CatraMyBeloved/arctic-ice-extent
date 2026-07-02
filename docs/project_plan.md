# **Sea Ice Extent Anomaly Forecasting – Learning-Focused Project Plan (Revised)**

## 1. **Overall Goal**

Build a complete pipeline for **Arctic sea ice extent anomaly forecasting** that emphasizes *learning* geospatial-temporal workflows, databases, data storage formats, and machine learning.
Performance is secondary; the main goal is to **understand tools, data, and methods**, and to properly document the journey.

---

## 2. **Learning Goals**

1. Work with geospatial-temporal data in Python
2. Create a GIS-capable database using PostgreSQL/PostGIS
3. Learn to work with **xarray**, **dask**, and **Zarr** cloud files
4. Learn to use **Parquet** for efficient feature storage
5. Perform exploratory data analysis (EDA) for geospatial-timeseries
6. Fit simple ML models on the data (linear, penalized, tree ensembles)
7. Implement time-series specific models with lagged features and backtesting
8. Prototype an LSTM on the dataset (learning exercise, not performance-driven)
9. Evaluate predictions with appropriate metrics for geospatial-timeseries

*Secondary goal*: develop a basic understanding of Arctic seasonal cycles and anomaly patterns.

---

## 3. **Data Sources**

### 3.1 NSIDC Sea Ice Index

* **Data format**: Pre-computed CSV tables from NSIDC G02135 v4.0
  * Pan-Arctic daily extent: `N_seaice_extent_daily_v4.0.csv`
  * Regional daily extent: `N_Sea_Ice_Index_Regional_Daily_Data_G02135_v4.0.xlsx`
  * Climatology (1981-2010): `N_seaice_extent_climatology_1981-2010_v4.0.csv`
* **Regions covered**: 14 Arctic regions (Baffin, Barents, Beaufort, Bering, Canadian Archipelago, Central Arctic, Chukchi, East Siberian, Greenland, Hudson, Kara, Laptev, Okhotsk, St. Lawrence) plus pan-arctic
* **Storage**: PostgreSQL tables with extent in million km² (Mkm²) units

### 3.2 ERA5 Reanalysis (Copernicus Climate Data Store)

* **Access method**: Downloaded as NetCDF files via CDS API
* **Variables**: 2m temperature (`t2m`), mean sea-level pressure (`msl`), 10m u-wind (`u10`), 10m v-wind (`v10`), total precipitation (`tp`), derived wind speed
* **Processing pipeline**:
  * Download monthly NetCDF files by variable
  * Merge variables and apply unit conversions (K→°C, Pa→hPa)
  * Aggregate to regional statistics (mean, std, p15, p85) using NSIDC region shapefiles
* **Storage**: Regional aggregations → **Parquet** (long format: date, region, variable, stat_type, value)
* **Regions**: 15 total (14 Arctic regions + Pan-Arctic aggregate)

---

## 4. **System Architecture**

* **PostgreSQL Database**

  * Tables:
    * `ice_extent_pan_arctic_daily`: Daily pan-Arctic extent (date, region, extent_mkm2)
    * `ice_extent_regional_daily`: Daily regional extent (date, region, extent_mkm2)
    * `ice_extent_climatology`: Day-of-year climatology with percentiles (dayofyear, avg_extent, std_dev, p10, p25, p50, p75, p90)
      * **Note**: Current implementation uses all available years (1979-2023) for climatology computation vs stated 1981-2010 WMO standard. See `docs/data_dictionary.md` for details.
  * Primary keys: `(region, date)` for efficient time-series queries
  * Units: All extent values in million km² (Mkm²)

* **Parquet Feature Store**

  * Schema: `date, region, variable, stat, value`
  * Files: `data/processed/parquet/era5_regional_{year}.parquet`
  * Partitioned by year; stored locally
  * Contains regional atmospheric statistics for all ERA5 variables

* **Data Pipeline**

  * NSIDC: CSV/Excel → PostgreSQL (direct ingestion, no GIS processing)
  * ERA5: CDS API → NetCDF (raw) → NetCDF (interim, merged/transformed) → Parquet (aggregated by region)
  * Unified access via `src/data_utils.load_data()` function

---

## 5. **Implementation Phases**

### Phase 0 – Warmup (Completed)

* NSIDC CSV data loaded into PostgreSQL
* ERA5 data download and transformation pipeline established
* Basic data access through `load_data()` utility function

### Phase 1 – Data Pipeline (Completed)

1. **NSIDC ingestion → PostgreSQL**

   * Full historical data (1979-2023) ingested from NSIDC CSV/Excel files
   * Pan-Arctic and regional daily extent stored in separate tables
   * Climatology baseline (1981-2010) loaded with percentile data

2. **ERA5 download and processing**

   * CDS API download script for monthly NetCDF files (1979-2023)
   * Transformation pipeline: variable merging, unit conversions, wind speed derivation
   * Regional aggregation using NSIDC shapefiles with regionmask
   * Pan-Arctic aggregation from combined regional masks

3. **Feature storage → Parquet**

   * Regional atmospheric statistics (mean, std, p15, p85) stored in yearly Parquet files
   * Long-format schema enables flexible querying by region/variable/stat
   * Partitioned by year for efficient access

### Phase 2 – Exploratory Analysis (Completed)

* Time series visualizations of ice extent and atmospheric variables
* Seasonal cycle heatmaps by region
* Temperature-ice extent correlation analysis with seasonal breakdown
* Climatology computation for all regions and variables
* Trend analysis using linear regression

### Phase 3 – Baseline Modeling (In Progress)

1. **Climatology baseline**

   * Day-of-year mean climatology computed for all regions
   * Used as benchmark for anomaly calculations

2. **Persistence baseline** (Completed)

   * Daily persistence (y_t+1 = y_t) evaluated on 2020-2023 in `03b_baseline_models.ipynb`
   * RMSE ≈ 0.087 Mkm² — the hard bar to beat at daily scale

3. **SARIMA models** (Completed)

   * Monthly aggregation of daily data to make SARIMA tractable
   * Model 1: SARIMA(1,0,1)×(0,1,1,12) on raw extent values
   * Model 2: SARIMA(2,0,2)×(1,0,1,12) on anomaly values
   * Train 1989-2019 / test 2020-2023 (48 months), **1-month-ahead walk-forward**
   * Performance: SARIMA_raw RMSE ≈ 0.227 Mkm² (skill vs persistence +0.88, vs climatology +0.72)

4. **Simple ML models** (Pending)

   * Linear regression, Ridge, Lasso
   * Random Forest, XGBoost
   * Multi-horizon predictions (+7, +14, +30 days)

### Phase 4 – Neural Network Experiments

**Implementation Status** (What has been built):

* **Basic LSTM (Univariate)**
  * [x] Architecture implemented: 2-layer LSTM (64 hidden units) + dropout (0.2)
  * [x] Training pipeline: Early stopping (patience=15), gradient clipping, LR scheduling
  * [x] Data split: 1989-2019 training, 2020-2023 test
  * [x] Training completed: Best validation loss ~0.002006 (normalized MSE)

* **Multivariate LSTM Variants**
  * [x] Non-lagged variant (7 features): extent + ERA5 means/stds
    * Best validation loss: ~0.000323 (normalized MSE)
  * [x] Lagged variant (13 features): base features + extent/temperature lags (t-7, t-14, t-30)
    * Best validation loss: ~0.000306 (normalized MSE)
  * [x] Cyclical variant (9 features): base features + day-of-year sin/cos encoding
    * Best validation loss: ~0.000321 (normalized MSE)

* **Seq2Seq LSTM (Multi-Horizon)**
  * [x] Vanilla encoder-decoder architecture (no attention, no teacher forcing)
  * [x] 7-day forecast horizon (30-day input → 7-day output)
  * [x] Multiple variants trained: univariate, multivariate, cyclical
    * Univariate: ~0.001419 validation loss
    * Multivariate: ~0.001631 validation loss

**Rebuilt (this session)**: The LSTM code was consolidated into `src/lstm_utils.py`
(shared datasets/model/training loop), fixing a **test-set leakage bug** — the old
notebooks used the 2020-2023 test set for early stopping and checkpoint selection.
Training now uses a **three-way temporal split**: train 1989-2014, validation
2015-2019 (model selection), test 2020-2023 (held out). Runs are seeded, save
self-contained checkpoint bundles (weights + scaler + config + split), and the
data is auto-bootstrapped via `src/data_bootstrap.ensure_extent_data()`.

**Validation Status** (What has been evaluated):

* [x] Training convergence verified (all models reached early stopping)
* [x] **Denormalization to Mkm²** for fair comparison (built into evaluation)
* [x] **Evaluation against persistence & climatology baselines** — univariate LSTM
  (04) done: RMSE 0.073 Mkm², beats persistence (skill +0.168), logged to
  `results/model_comparison.csv`
* [x] **Seasonal performance breakdown** (winter vs summer) for 04
* [ ] Multivariate (05) and Seq2Seq (06) evaluated — **refactored & ready, pending a GPU-box run** (need ERA5 parquet)
* [ ] **Statistical significance testing** (Diebold-Mariano) — in `07_model_comparison.ipynb`
* [ ] Feature ablation significance (05 variants loop provides the comparison; significance pending)
* [ ] **Comprehensive lessons documented**

**Important Notes**:

* **Leakage fixed**: previous results selected the model on the test set; all new
  results use a held-out validation era, so they are comparable to the baselines.
* **Compute is not the bottleneck at this stage**: these 1-D models train in
  minutes even on CPU. The 4060's headroom is best spent on sweeps/seeds/ensembles
  now, and on the spatial CNN-LSTM (Phase 6) later.

**Next Steps**:

1. ✅ Evaluation framework built (`src/evaluation_utils.py`) and baselines logged (03a/03b)
2. ✅ LSTM infrastructure rebuilt (`src/lstm_utils.py`, `src/train.py`); univariate (04) evaluated
3. Run multivariate (05) and seq2seq (06) on the GPU box to fill in their rows
4. Comprehensive comparison + significance in `notebooks/07_model_comparison.ipynb`

---

### Phase 4.1 – Uncertainty Quantification & Extended Horizons

**Implementation Status**: Components 1-2 completed on the univariate model (laptop, no ERA5
needed); rerun on the multivariate/lagged variant once 05/06 have GPU-box results. Component 3
scoped (autoregression dropped, see below). Component 4 still planned.

**Goal**: Extend LSTM experiments to quantify prediction uncertainty and evaluate longer forecast horizons using ensemble methods, dropout-based uncertainty, and autoregressive architectures.

**Components**:

1. **LSTM Ensemble (10 Models)** — `09_lstm_ensemble.ipynb` (Completed, univariate)
   * Train 10 independent LSTM models with different random initializations
   * Use best-performing architecture from Phase 4 (multivariate lagged variant) — run on the
     univariate model for now since it needs no ERA5 data; rerun on 05's best variant later
   * Generate prediction intervals from ensemble spread (mean, std, percentiles)
   * Compare ensemble mean vs single-model performance
   * Analyze inter-model variance as proxy for epistemic uncertainty
   * Document computational costs and convergence patterns across ensemble members
   * **Result**: member RMSE 0.0602 ± 0.0004 (tight — 10 seeds converge to nearly the same
     function), ensemble mean RMSE 0.0597, skill vs persistence +0.316 (significant, DM p ≈ 0).
     Ensemble beats a *typical* member (DM p = 0.022) but not every member individually. **90%
     interval PICP only 0.170** (badly overconfident) — epistemic (inter-seed) spread massively
     underestimates real error here, since all seeds land in a near-identical solution.

2. **MC Dropout for Uncertainty Estimation** — `10_mc_dropout.ipynb` (Completed, univariate)
   * Implement Monte Carlo Dropout during inference (dropout enabled at test time)
   * Generate N forward passes per input (e.g., N=50 or N=100)
   * Compute prediction statistics: mean, standard deviation, confidence intervals
   * Compare MC Dropout uncertainty estimates vs ensemble uncertainty
   * Analyze uncertainty calibration (reliability diagrams)
   * Evaluate whether high-uncertainty predictions correlate with higher errors
   * **Result**: added a `head_dropout` param to `IceExtentLSTM` for this purpose. With
     `head_dropout=0.3`, point-forecast RMSE got *worse* (0.095, skill vs persistence **-0.089** —
     not significantly different from persistence, DM p = 0.136), evidently too much regularization
     for this small a hidden layer. **90% interval PICP 0.938** (close to nominal, unlike the
     ensemble) but **MPIW ~36x wider** than the ensemble's — well-covered mostly by being very wide,
     not by being precisely calibrated. Combined with Component 1: neither method's uncertainty
     correlates strongly with actual error (corr ≈ 0.15 and 0.04 respectively), suggesting this
     model's error is mostly aleatoric, not epistemic — a genuinely calibrated interval would likely
     need a distribution-modeling approach (e.g. quantile regression) rather than more UQ tuning.

3. **Direct Multi-Horizon Seq2Seq for Extended Horizons** (autoregression dropped — see below)

   **Why this component matters now, precisely:** the residual-diagnostics dive after Components
   1-2 found that the univariate 1-day model's RMSE (0.0593 Mkm²) sits almost exactly on NSIDC's
   own documented daily-extent measurement uncertainty (~0.06 Mkm², 2σ) — see `results/` /
   `07_model_comparison.ipynb` findings. Further 1-day-horizon work has limited headroom: we're
   close to the precision limit of the data itself, not the model. Extended horizons are where the
   open scientific questions actually are now — chasing a data-noise floor at 1 day is a lower
   priority than mapping how far into the future the model still beats trivial baselines at all.

   **Goal**: build the **skill-decay curve** — RMSE/skill vs persistence & climatology across lead
   time — and find the crossover lead time where skill drops to ≤0 (the point past which the model
   is no more useful than a trivial baseline). Also check whether short-lead-time RMSE (1-3 days,
   read off the shortest-horizon model) still sits near the ~0.06 Mkm² floor, or whether climate
   features (05/06) move it — that's the test for whether the daily floor was data-precision-limited
   or information-limited.

   **Separate models per target horizon, not one shared-output model.** Extend `forecast_horizon`
   to 14-day and 30-day as their own dedicated training runs (own `IceExtentLSTM` instance, own
   `train_model` call each), alongside the existing 7-day model from `06_seq2seq_lstm` — do **not**
   train one model with `forecast_horizon=30` and read off day-7/day-14 from its output. This is the
   classic **"direct" vs "MIMO" multi-horizon tradeoff** in time-series forecasting (Taieb &
   Hyndman): a single model trained with one aggregate loss across a 30-day output has to compromise
   across horizons of very different difficulty (day-1 easy, day-30 hard) with no way to protect the
   easy ones from the hard ones' gradient signal. A model trained end-to-end for exactly one horizon
   has no such conflict. (A single wide-output MIMO model vs these dedicated models is itself a good
   later ablation, if the tradeoff turns out to matter in practice — but dedicated models are the
   default here.)
   * Train dedicated models at `forecast_horizon` = 14 and 30 (7 already exists)
   * Build the skill-decay curve across 1/7/14/30-day lead times from the three models together
   * Analyze error growth over forecast lead time (RMSE/skill curve vs day-ahead)
   * Evaluate uncertainty growth with forecast lead time (pairs with ensemble/MC
     Dropout from Components 1-2)

   **Why not autoregressive:** an autoregressive loop (feed the t+1 prediction back
   in as input for t+2) only works cleanly for the univariate model. For the
   multivariate/climate-feature models it silently requires *future* climate
   inputs at every step beyond the first — data that doesn't exist for a genuine
   future forecast (ERA5 is reanalysis, not a forecast product). We looked at how
   published sea ice forecasters solve this and none of them solve it this way:
   IceNet (Andersson et al. 2021, *Nat. Commun.*) predicts all 6 of its future
   months in a single forward pass from historical-only input channels — it never
   feeds a predicted month back in, and doesn't even use CMIP6-simulated future
   atmosphere as forcing. The dynamical alternative (ECMWF's SEAS5) solves it
   differently again, by coupling atmosphere/ocean/ice into one jointly-integrated
   model rather than sourcing an external forecast. Direct multi-horizon output
   from historical inputs only is the approach both traditions converge on, so
   that's what we do too — extend `forecast_horizon`, not roll predictions forward.
   (A real future-forcing pipeline, e.g. ECMWF Open Data / `ecmwf-opendata` for the
   HRES forecast, stays a possible follow-on for genuine operational deployment
   later, but isn't required for hitting the extended-horizon goal.)

4. **Enhanced Encoder-Decoder LSTM**
   * Implement attention mechanism for Seq2Seq architecture
   * Add teacher forcing during training (scheduled sampling)
   * Test bidirectional encoder for improved context modeling
   * Multi-horizon outputs: 7-day, 14-day, 30-day forecast sequences
   * Compare enhanced Seq2Seq vs vanilla version from Phase 4
   * Analyze attention weights to understand temporal dependencies

5. **Predictive VAE (Variational Seq2Seq)** — `13_predictive_vae.ipynb` (Planned)

   **Motivation, precisely:** Components 1-2 found that both the 10-seed ensemble and MC Dropout
   correlate weakly with actual error (corr ≈ 0.04 / 0.15 — see their results above) because they
   measure *epistemic* uncertainty (uncertainty about which weights are right), while the
   residual-diagnostics dive after them (Ljung-Box + `corr(|residual|, |actual day-to-day change|)
   = 0.456`) pointed at the dominant error source being *aleatoric* — genuine day-to-day
   unpredictability the model can't resolve from its own history, concentrated on high-volatility
   days. A VAE's latent variable is trained explicitly to model a *distribution* of plausible
   futures given the same context, rather than one point estimate plus a weight-space proxy for
   uncertainty. Sampling `z` at inference gives a genuinely generative ensemble of trajectories —
   a structurally different (and better-motivated) way to capture aleatoric spread than dropout
   noise or inter-seed variance. Runs on the univariate + cyclical-time inputs, so needs no ERA5.

   **Architecture:**
   * **Encoder**: `nn.LSTM` over the 30-day input window (features: `extent_mkm2`,
     `day_of_year_sin`, `day_of_year_cos`) → final hidden state `h_T` (size `hidden_size`) →
     two linear heads `mu = Linear(hidden_size, latent_dim)`, `logvar = Linear(hidden_size,
     latent_dim)` → reparameterize: `z = mu + exp(0.5*logvar) * eps`, `eps ~ N(0, I)`.
     Start with `latent_dim` small (8-16) relative to `hidden_size=64` — a bottleneck is the point.
   * **Decoder**: a second `nn.LSTM`, initial hidden state set from `z` (e.g. `Linear(latent_dim,
     hidden_size)`), stepped autoregressively across the forecast horizon. **Autoregression is
     fine here specifically because the only per-step decoder input is `day_of_year_sin/cos` for
     that future date — calendar arithmetic, always exactly knowable in advance.** This is the
     same distinction Phase 4.1 Component 3 draws for why autoregression is *not* fine for the
     climate-feature models: the problem was never autoregression itself, only feeding forward
     something not actually knowable (future weather). Each decoder step outputs a scalar extent
     prediction via a final `Linear(hidden_size, 1)` head.
   * **Loss (ELBO)**: `reconstruction_loss + beta * kl_loss`, where `reconstruction_loss` is
     MSE (or Gaussian NLL, if predicting per-step variance too — a natural stretch goal, since
     that would give a second, complementary uncertainty estimate) between the decoded sequence
     and the true future window, and `kl_loss = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))` is
     the standard closed-form KL divergence against a unit Gaussian prior.
   * **Inference modes**: `mu` directly (no sampling) → point forecast, evaluated in `07`'s table
     exactly like every other model. Sample `z` N times (e.g. N=50, matching MC Dropout's pass
     count for a fair comparison) → probabilistic ensemble, evaluated with the same
     `picp`/`mpiw`/`reliability_curve` helpers already built for 09/10.

   **Training details to get right:**
   * **KL annealing is not optional here** — ramp `beta` linearly from 0 to its target value over
     the first ~30-50% of training (a `kl_weight_warmup_epochs` config field). Starting at full
     `beta` risks posterior collapse (the decoder ignores `z` entirely, especially plausible given
     the "regression to the mean" tendency already observed in the plain LSTM) before the encoder
     has learned anything useful to put in `z`.
   * Watch `kl_loss` during training: if it collapses to ~0 early and stays there, that's collapse
     — mitigations in order of effort: lower `beta`, add free-bits (clamp per-dimension KL to a
     minimum), or feed `z` into *every* decoder step (concatenated with the day-of-year input)
     instead of only as the initial hidden state, which makes it harder for the decoder to ignore.
   * `latent_dim` and `beta` are the two new hyperparameters to sweep; keep `hidden_size`/
     `num_layers` at the same starting points as `06`/`11` rather than tuning everything at once.

   **Infra additions needed (small, scoped):**
   * `load_checkpoint` in `lstm_utils.py` currently hardcodes rebuilding an `IceExtentLSTM`
     regardless of the saved `model_class` field — needs a dispatch fix (e.g. a small registry
     dict) before a second model class can round-trip through checkpoints.
   * New `PredictiveVAE` model class (separate from `IceExtentLSTM`, doesn't belong in the same
     class — different forward signature, different loss) and a dedicated training loop (own ELBO
     loss, not `train_model`'s plain MSE) — natural home is a new `src/vae_utils.py` mirroring
     `lstm_utils.py`'s structure (`TrainConfig`-equivalent with `latent_dim`/`beta`/
     `kl_weight_warmup_epochs` added, `train_vae`, `sample_futures`), reusing `SequenceDataset`,
     `set_seed`, `get_device`, `save_checkpoint`/`load_checkpoint`, and evaluation utilities as-is.

   **Evaluation plan:**
   * Point-forecast RMSE/skill in `07_model_comparison.ipynb`'s table, same as every other model.
   * Probabilistic comparison against 09 (ensemble) and 10 (MC Dropout) on identical metrics: PICP/
     MPIW at 90%, reliability diagram (three-way overlay), `corr(uncertainty, |error|)`. This is the
     real test — does sampling `z` produce intervals that are both better-calibrated *and* more
     informative (higher error-correlation) than the two methods already tried?
   * Target horizon: start with 14-day (dedicated model from Component 3), extend to 30-day only if
     14-day training is stable (no posterior collapse, sane calibration) — don't parallelize the
     horizon and the new architecture risk at the same time.

**Evaluation Focus**:
* Uncertainty quantification metrics: prediction interval coverage, sharpness
* Calibration analysis: reliability diagrams, calibration error
* Multi-horizon skill: RMSE/MAE curves vs forecast lead time
* Ensemble diversity vs accuracy trade-offs
* Computational cost analysis (training time, inference speed)

---

### Phase 5 – Advanced Features (Longer-Term)

* Expand to regional models (Tier B)
* Add more ERA5 variables (winds, geopotential)
* Implement ice-edge band approach (Tier C)
* Advanced spatio-temporal modeling

---

### Phase 6 – Spatial-Temporal CNN-LSTM (Advanced Learning-Oriented)

**Implementation Status**: Planned

**Goal**: Incorporate gridded spatial data (ERA5 weather fields) into ice extent forecasting using CNN encoders to extract spatial features, followed by LSTM temporal modeling for scalar predictions.

**Architecture Pipeline**:

1. **Spatial ERA5 Preprocessing**
   * Download gridded ERA5 data (not regional aggregates): T2M, MSLP, SST, sea ice concentration
   * Regrid to Arctic Stereographic projection (EPSG:3411, Arctic-focused equal-area grid)
   * Downsample to computationally feasible resolution (64×64 or 128×128 grid)
   * Create multi-channel "image" time series: each timestep = [T2M, MSLP, SST, SIC] stack
   * Normalize each channel (per-variable standardization)
   * Store as zarr arrays or HDF5 for efficient sequential access

2. **CNN Spatial Encoder**
   * Input: (batch, time_steps, channels, height, width) - e.g., (32, 30, 4, 64, 64)
   * For each timestep: apply CNN encoder to extract spatial feature vector
   * Architecture options:
     * **ResNet-style encoder**: 3-4 convolutional blocks with residual connections
     * **Small VGG-style**: 3-5 conv layers with pooling, output 256-512 feature vector
   * Output: (batch, time_steps, feature_dim) - compressed spatial representation per timestep
   * Optionally: pre-train CNN on auxiliary task (e.g., predict SIC from weather fields)

3. **LSTM Temporal Decoder (Scalar Predictions)**
   * Input: CNN-encoded spatial features (time_steps × feature_dim)
   * 2-3 layer LSTM processes temporal sequence of spatial features
   * Output: scalar prediction(s) for pan-Arctic sea ice extent (Mkm²)
   * Multi-horizon variants:
     * Single-step: predict extent at t+1
     * Seq2Seq: predict extent for t+1 to t+7, extendable to 14-30 days as a direct
       multi-step output (no autoregressive rollout — see Phase 4.1 Component 3)

4. **Hybrid Architecture (CNN-LSTM + Tabular Features)**
   * Combine CNN-encoded spatial features with region-aggregated tabular features
   * Concatenate spatial feature vector with climatology, lagged extent, day-of-year
   * Pass combined features to LSTM for temporal modeling
   * Allows model to leverage both spatial patterns and pre-computed statistics

**Training Strategy**:

* **Modular pre-training**: Train CNN encoder separately on spatial prediction task (optional)
* **End-to-end training**: Jointly optimize CNN + LSTM on extent forecasting
* **Memory efficiency**: Use gradient checkpointing for CNN, sliding window batches for long sequences
* **Data augmentation**: Random crops, spatial shifts (if applicable to Arctic grid)
* **Regularization**: Spatial dropout in CNN, temporal dropout in LSTM

**Evaluation & Analysis**:

* **Performance comparison**: CNN-LSTM vs aggregated-feature LSTM from Phase 4/4.1
* **Ablation studies**:
  * CNN features only vs tabular features only vs hybrid
  * Spatial resolution impact: 64×64 vs 128×128 grids
  * Number of input channels: T2M only vs full 4-channel stack
* **Spatial interpretability**:
  * Visualize CNN activation maps (GradCAM or saliency maps)
  * Identify which Arctic regions drive predictions (spatial attention)
  * Analyze seasonal patterns in CNN feature importance
* **Computational cost analysis**:
  * Training time vs aggregated models (likely 5-10x slower)
  * Inference latency (important for operational forecasting)
  * Memory requirements and scalability

**Deliverables**:

* Gridded ERA5 preprocessing pipeline (zarr/HDF5 storage)
* CNN-LSTM implementation in PyTorch with modular components
* Trained models: CNN-only, LSTM-only, hybrid CNN-LSTM
* Evaluation notebook comparing spatial vs aggregated approaches
* Visualization of learned spatial features and attention patterns
* Documentation of computational trade-offs and optimization strategies

---

## 6. **Evaluation Strategy**

* Compare models against persistence and climatology baselines
* Use multiple metrics: RMSE, MAE, anomaly correlation coefficient
* Seasonal breakdown (evaluate winter vs summer separately)
* Document strengths/weaknesses per horizon and per model class

---

## 7. **Milestones**

**M0 – Warmup**

* [x] NSIDC data loaded into PostgreSQL
* [x] ERA5 download and transformation pipeline established
* [x] Data access utilities created

**M1 – Basic Pipeline**

* [x] Full historical NSIDC data (1979-2023) ingested
* [x] Pan-Arctic and regional extent tables populated
* [x] ERA5 data downloaded for all years (1979-2023)
* [x] Regional aggregations computed and stored in Parquet

**M2 – EDA**

* [x] Time series visualizations for extent and atmospheric variables
* [x] Seasonal cycle heatmaps by region
* [x] Temperature-ice extent correlation analysis
* [x] Climatology computation for all regions

**M3 – Baselines**

* [x] Climatology baseline computed
* [x] SARIMA models trained and evaluated (monthly data)
* [x] Persistence baseline (daily data) — RMSE ≈ 0.087 Mkm²
* [ ] Simple ML models (Linear, Ridge, Random Forest, XGBoost)
* [ ] Multi-horizon predictions (+7, +14, +30 days)

**M4 – Time-Series Feature Engineering**

**Implementation**:
* [x] Lagged features implemented in LSTM (t-7, t-14, t-30 for extent and temperature)
* [x] Seasonal encoding implemented (sin/cos of day-of-year)

**Validation**:
* [ ] Lagged features validated via ablation studies (test impact of removing each lag)
* [ ] Seasonal encoding validated (compare with vs without cyclical features)
* [ ] Expanding-window backtesting framework created
* [ ] Multi-horizon models with lagged features (simple ML: Linear, RF, XGBoost)

**M5 – LSTM Experiments**

**Implementation**:
* [x] Shared engine `src/lstm_utils.py` (datasets, model, seeded/AMP training, 3-way split, checkpoint bundles)
* [x] Headless CLI `src/train.py` and data bootstrap `src/data_bootstrap.py`
* [x] Basic (04), multivariate (05), seq2seq (06) notebooks refactored onto the engine
* [x] Test-set leakage fixed via held-out validation era (2015-2019)

**Validation**:
* [x] Training convergence verified (validation loss tracking)
* [x] **Denormalization and metric standardization** (Mkm²) — built into evaluation
* [x] **Evaluation vs baselines** for univariate (04); 05/06 pending GPU run
* [x] **Seasonal performance breakdown** (winter vs summer) for 04
* [x] **Statistical significance** (Diebold-Mariano + Holm-Bonferroni) — in `07_model_comparison.ipynb`
* [ ] **Multi-horizon error analysis** (Seq2Seq day 1-7) — 06 pending GPU run
* [x] **Model comparison** (all models, identical metrics) — `07_model_comparison.ipynb`
* [x] **Time-series backtesting** (5-fold expanding window) — `08_time_series_backtesting.ipynb`,
  univariate LSTM beats persistence in 5/5 folds
* [ ] **Lessons documented** (what worked, what didn't, why)

**M5.1 – Uncertainty Quantification & Extended Horizons**

**Ensemble Models** (Completed, univariate — `09_lstm_ensemble.ipynb`):
* [x] Train 10-model LSTM ensemble with different initializations
* [x] Implement ensemble prediction statistics (mean, std, percentiles)
* [x] Evaluate ensemble mean vs single-model performance — mean RMSE 0.0597 vs member mean
  0.0602 ± 0.0004; beats a typical member (p=0.022) but not the single best member
* [x] Analyze epistemic uncertainty from ensemble spread — 90% PICP only 0.170 (badly
  overconfident: seeds converge to nearly the same function)
* [x] Document computational costs across ensemble members — ~223s/member on CPU, 2227s total for 10

**MC Dropout** (Completed, univariate — `10_mc_dropout.ipynb`):
* [x] Implement MC Dropout inference (50-100 forward passes) — added `head_dropout` to `IceExtentLSTM`
* [x] Generate prediction intervals from dropout sampling
* [x] Compare MC Dropout vs ensemble uncertainty estimates — MC Dropout PICP 0.938 but MPIW ~36x wider
* [x] Analyze uncertainty calibration (reliability diagrams)
* [x] Evaluate high-uncertainty prediction correlation with errors — weak for both (0.15 / 0.04)

**Extended-Horizon Seq2Seq** (dedicated direct model per horizon, no autoregression, no shared
MIMO output — see Phase 4.1 Component 3):
* [ ] Train dedicated `forecast_horizon=14` model
* [ ] Train dedicated `forecast_horizon=30` model
* [ ] Build the skill-decay curve (1/7/14/30-day) and find the zero-skill crossover lead time
* [ ] Analyze error growth over extended horizons
* [ ] Document uncertainty growth with forecast lead time

**Enhanced Encoder-Decoder**:
* [ ] Add attention mechanism to Seq2Seq LSTM
* [ ] Implement teacher forcing with scheduled sampling
* [ ] Test bidirectional encoder architecture
* [ ] Multi-horizon outputs (7-day, 14-day, 30-day)
* [ ] Analyze attention weights for temporal dependencies

**Predictive VAE** (Variational Seq2Seq — see Phase 4.1 Component 5 for full spec):
* [ ] Fix `load_checkpoint` model-class dispatch (currently hardcodes `IceExtentLSTM`)
* [ ] Implement `src/vae_utils.py`: `PredictiveVAE` model, ELBO loss, KL annealing, `train_vae`,
  `sample_futures`
* [ ] Train 14-day model with KL annealing; verify no posterior collapse (`kl_loss` not ~0)
* [ ] Point-forecast evaluation in `07_model_comparison.ipynb`'s table
* [ ] Probabilistic comparison vs 09 (ensemble) and 10 (MC Dropout): PICP/MPIW @ 90%, reliability
  diagram, `corr(uncertainty, |error|)`
* [ ] Extend to 30-day only once 14-day is stable and calibration results are sane

**M6 – Advanced Features**

* [ ] Regional models for all 14 Arctic regions
* [ ] Additional ERA5 variables (geopotential, longwave radiation)
* [ ] Cross-regional feature engineering
* [ ] Expanded evaluation across regions

**M7 – Spatial-Temporal CNN-LSTM**

**Data Preparation**:
* [ ] Download gridded ERA5 data (T2M, MSLP, SST, SIC)
* [ ] Regrid to Arctic Stereographic projection (EPSG:3411)
* [ ] Downsample to 64×64 and 128×128 grids
* [ ] Create multi-channel image time series
* [ ] Store as zarr/HDF5 for efficient access

**Model Implementation**:
* [ ] CNN spatial encoder implemented (ResNet or VGG-style)
* [ ] LSTM temporal decoder for scalar predictions
* [ ] Hybrid CNN-LSTM + tabular features architecture
* [ ] End-to-end training pipeline with gradient checkpointing
* [ ] Multi-horizon prediction variants (single-step, Seq2Seq, autoregressive)

**Evaluation & Analysis**:
* [ ] Performance comparison vs aggregated-feature LSTM
* [ ] Ablation studies (CNN-only, tabular-only, hybrid)
* [ ] Spatial resolution impact analysis (64×64 vs 128×128)
* [ ] CNN activation visualization (GradCAM/saliency maps)
* [ ] Computational cost documentation (training time, memory, inference)
* [ ] Spatial interpretability analysis completed

---

## 8. **Notebook Structure (Current Implementation)**

**Data Ingestion & Processing**

1. **01a\_data\_ingestion\_era5\_download.ipynb**
   * Downloads ERA5 data from Copernicus CDS API
   * Monthly NetCDF files by variable (1979-2023)
   * Implements retry logic with exponential backoff

2. **01b\_data\_ingestion\_nsidc.ipynb**
   * Loads NSIDC CSV/Excel files into PostgreSQL
   * Creates tables: `ice_extent_pan_arctic_daily`, `ice_extent_regional_daily`, `ice_extent_climatology`
   * Handles region name mapping and unit conversions (km² → Mkm²)

3. **01c\_data\_ingestion\_era5\_transformation.ipynb**
   * Merges monthly ERA5 variable files
   * Applies unit transformations (K→°C, Pa→hPa)
   * Computes derived variables (wind speed from u/v components)
   * Aggregates to Arctic regions using NSIDC shapefiles + regionmask
   * Saves regional statistics to yearly Parquet files

**Analysis & Modeling**

4. **02\_EDA.ipynb**
   * Time series visualizations (extent + atmospheric variables)
   * Seasonal cycle heatmaps by region
   * Temperature-ice extent correlation with seasonal coloring
   * Trend analysis using linear regression

5. **03a\_sarima\_baseline.ipynb** (Completed)
   * Two SARIMA models on monthly aggregated extent (raw + anomaly), loaded direct from the DB
   * **1-month-ahead walk-forward** evaluation (train 1989-2019 / test 2020-2023, 48 months)
   * Logged to `results/model_comparison.csv` at `scale="monthly"` alongside monthly baselines
   * Key result: SARIMA_raw RMSE ≈ 0.227 Mkm² (skill vs persistence +0.88, vs climatology +0.72)

6. **03b\_baseline\_models.ipynb** (Completed)
   * Persistence (y\_t+1 = y\_t) and climatology (day-of-year mean) baselines evaluated on the
     2020-2023 daily pan-Arctic test set using `src/evaluation_utils.py`
   * Results logged to `results/model_comparison.csv`; seasonal breakdown + figure produced
   * Key result: persistence RMSE ≈ 0.087 Mkm² (the bar to beat); climatology ≈ 1.0 Mkm²

7. **04\_basic\_lstm.ipynb** (Completed & evaluated)
   * Univariate LSTM (extent only), thin narrative over `src/lstm_utils.py`
   * Three-way split (train 1989-2014 / val 2015-2019 / test 2020-2023), seeded
   * Denormalized evaluation vs persistence & climatology, seasonal breakdown, figure
   * **Result**: RMSE 0.073 Mkm², beats persistence (skill +0.168); logged as `LSTM_Basic_Univariate` (daily)

8. **05\_multivariate\_lstm.ipynb** (Refactored — pending GPU run)
   * Adds ERA5 climate features; a DRY variants loop trains and compares
     `climate`, `climate+cyclical`, `climate+lags` through the shared engine
   * Same three-way split and denormalized baseline comparison as 04
   * Needs the ERA5 parquet store → run on the GPU box; logs best variant (daily)

9. **06\_seq2seq\_lstm.ipynb** (Refactored — pending GPU run)
   * Multi-day (7-day) forecasting: `forecast_horizon=7` on the same engine
   * Compares univariate / climate / climate+cyclical, plots error-by-forecast-day
   * Needs ERA5 parquet → run on the GPU box; logs best variant at `scale="multistep_7d"`

**Experiments**

10. **experiments/shapefiles.ipynb**
    * Shapefile exploration and region definitions
    * NSIDC region visualization

**Planned Notebooks** (High Priority for Scientific Rigor)

11. **07\_model\_comparison.ipynb** (Completed — self-updating)
    * Reads `results/model_comparison.csv` → ranked comparison table per scale
    * Regenerates daily 1-step predictions from every checkpoint bundle in `models/` that shares
      the fixed 2020-2023 test era (excludes 08's per-fold bundles by design — see below)
    * **Diebold-Mariano significance** vs persistence & climatology, **Holm-Bonferroni** correction
      across the full family of tests once more than one bundle is present (both in
      `evaluation_utils`)
    * Skill-score chart, seasonal breakdown, and per-year error breakdown of the best daily model
    * Laptop result: every univariate LSTM variant beats persistence significantly except MC
      Dropout (DM ≈ -9.4 to -16.9, p ≈ 0, survives Holm-Bonferroni); best daily model is
      `09_ensemble_seed8` (RMSE 0.0593). 2020 flagged as the highest-error test year, independently
      confirmed by notebook 08's backtest. Rerun on the GPU box and the 05/06 rows fill in
      automatically.

13. **08\_time\_series\_backtesting.ipynb** (Completed, univariate)
    * Expanding-window cross-validation: 5 folds (test years 2019-2023), each with its own 2-year
      validation window and all prior history as training data
    * Retrains the univariate architecture per fold, aggregates RMSE/skill with mean ± std
    * **Result**: LSTM beats persistence in **5/5 folds** — RMSE 0.0623 ± 0.0102 vs persistence's
      0.0882 ± 0.0036, skill +0.296 ± 0.088 (higher than the single fixed-split result in 04,
      +0.168 — likely more training data per fold). 2020 is consistently the weakest fold.
    * Caught and fixed a real methodology bug in `07_model_comparison.ipynb`: its checkpoint
      auto-discovery originally re-evaluated these fold bundles on the shared 2020-2023 window,
      silently reusing each fold's own validation years as test data. Fixed by excluding
      `08_backtest_fold_*` bundles from 07's shared-test-era comparison by name.

**Planned Notebooks** (Lower Priority)

12. **05\_ml\_baselines.ipynb** (Planned)
    * Linear regression, Ridge, Lasso, Random Forest, XGBoost
    * Multi-horizon predictions (+7, +14, +30 days)
    * Skill scores vs persistence and climatology

**Completed Notebooks** (Phase 4.1 - Uncertainty & Extended Horizons)

14. **09\_lstm\_ensemble.ipynb** (Completed, univariate)
    * Train 10 LSTM models with different random seeds
    * Ensemble prediction statistics (mean, std, percentiles)
    * Epistemic uncertainty analysis from ensemble spread
    * Comparison: ensemble mean vs best single model
    * Computational cost analysis
    * **Result**: member RMSE 0.0602 ± 0.0004 (10 seeds converge to nearly the same function),
      ensemble mean RMSE 0.0597, skill vs persistence +0.316. Ensemble beats a *typical* member
      (DM p = 0.022) but not the single best member. **90% interval PICP only 0.170** — badly
      overconfident, since epistemic spread across seeds is tiny relative to real error.

15. **10\_mc\_dropout.ipynb** (Completed, univariate)
    * Implement MC Dropout inference (50-100 forward passes); added a `head_dropout` param to
      `IceExtentLSTM` since between-layer LSTM dropout alone doesn't reach the output layer
    * Generate prediction intervals and uncertainty estimates
    * Reliability diagrams and calibration analysis
    * Compare MC Dropout vs ensemble uncertainty (reuses 09's saved checkpoints, no retraining)
    * Correlation between uncertainty and prediction errors
    * **Result**: `head_dropout=0.3` cost real point-forecast accuracy (RMSE 0.095, skill
      **-0.089** vs persistence — not significant, DM p = 0.136). **90% interval PICP 0.938**
      (much better covered than the ensemble) but **MPIW ~36x wider** — good coverage mostly from
      being very wide, not precise calibration. Both methods' uncertainty correlates weakly with
      actual error (0.15 / 0.04), suggesting the dominant error source here is aleatoric, not
      epistemic — neither ensembling nor dropout targets that directly.

16. **11\_extended\_horizon\_seq2seq.ipynb** (Planned)
    * Two dedicated direct models, `forecast_horizon=14` and `forecast_horizon=30`, each trained
      end-to-end for its own horizon (not one shared-output model sliced afterward — see Phase 4.1
      Component 3 for the direct-vs-MIMO reasoning; also no autoregressive rollout, same section)
    * Skill-decay curve across 1/7/14/30-day lead times (stitching in 04's 1-day and 06's 7-day
      results), and the lead time where skill vs persistence/climatology crosses zero
    * Error growth analysis over forecast lead time
    * Uncertainty growth with horizon length

17. **12\_attention\_seq2seq.ipynb** (Planned)
    * Enhanced encoder-decoder with attention mechanism
    * Teacher forcing with scheduled sampling
    * Bidirectional encoder experiments
    * Multi-horizon outputs (7-day, 14-day, 30-day)
    * Attention weight visualization and interpretation

18. **13\_predictive\_vae.ipynb** (Planned)
    * Variational Seq2Seq: LSTM encoder → `(mu, logvar)` → reparameterized `z` → LSTM decoder,
      autoregressive over the known future day-of-year (see Phase 4.1 Component 5 for the full
      architecture, ELBO loss, KL-annealing plan, and infra prerequisites)
    * Point forecast from `mu`; probabilistic ensemble from N samples of `z`
    * Three-way calibration comparison against 09 (ensemble) and 10 (MC Dropout) using the same
      `picp`/`mpiw`/reliability-diagram harness
    * Target: 14-day first (reusing 11's dedicated-horizon setup), 30-day only once stable

**Planned Notebooks** (Phase 6 - CNN-LSTM)

19. **14\_era5\_spatial\_preprocessing.ipynb** (Planned)
    * Download gridded ERA5 data (T2M, MSLP, SST, SIC)
    * Regrid to Arctic Stereographic (EPSG:3411)
    * Downsample to 64×64 and 128×128 grids
    * Create multi-channel image sequences
    * Save to zarr/HDF5 format

20. **15\_cnn\_encoder.ipynb** (Planned)
    * CNN spatial encoder implementation (ResNet/VGG-style)
    * Feature extraction from gridded weather data
    * Optional pre-training on auxiliary task
    * Spatial feature visualization

21. **16\_cnn\_lstm\_hybrid.ipynb** (Planned)
    * Full CNN-LSTM pipeline: spatial encoding + temporal modeling
    * Hybrid architecture (CNN features + tabular features)
    * Multi-horizon prediction variants
    * Training with gradient checkpointing

22. **17\_cnn\_lstm\_evaluation.ipynb** (Planned)
    * Performance comparison vs aggregated-feature LSTM
    * Ablation studies (CNN-only, tabular-only, hybrid)
    * Spatial resolution impact (64×64 vs 128×128)
    * GradCAM/saliency map visualization
    * Regional importance and seasonal pattern analysis
    * Computational cost analysis (training time, memory, inference)

---

## 9. **Success Criteria**

**Data Infrastructure** (Completed):
* ✅ Working PostgreSQL + Parquet data pipeline
* ✅ Full historical data ingested (1979-2023)
* ✅ Regional aggregation framework implemented
* ✅ Exploratory data analysis completed with visualizations

**Model Implementation** (Completed):
* ✅ SARIMA baseline models trained (raw + anomaly variants)
* ✅ Basic LSTM implemented and trained (univariate)
* ✅ Multivariate LSTM variants implemented (non-lagged, lagged, cyclical)
* ✅ Seq2Seq LSTM implemented (7-day multi-horizon)

**Evaluation & Validation** (In Progress - CRITICAL for Scientific Rigor):
* ✅ SARIMA evaluated with RMSE/MAE/MAPE in Mkm²
* ✅ Baseline models evaluated (persistence, climatology) on the 2020-2023 daily test set via `03b_baseline_models.ipynb`; results in `results/model_comparison.csv` (persistence RMSE ≈ 0.087 Mkm², climatology ≈ 1.0 Mkm²)
* ✅ Evaluation framework created (`src/evaluation_utils.py`): denormalization, metrics (RMSE/MAE/MAPE/skill score/ACC), baseline models, expanding-window backtesting, and results logging/comparison utilities
* ✅ LSTM predictions denormalized to Mkm² (built into `src/lstm_utils` evaluation)
* ✅ Univariate LSTM (04) evaluated vs baselines on the held-out test set (beats persistence)
* ✅ Seasonal performance breakdown (winter vs summer) for 04
* ⏳ Multivariate (05) & seq2seq (06) evaluated — refactored, pending GPU-box run
* ✅ All models on one table with significance testing (Diebold-Mariano + Holm-Bonferroni) —
  `07_model_comparison.ipynb` (univariate family; 05/06 rows pending GPU-box run)
* ⏳ Feature validation via ablation studies (05 variants loop provides the comparison, pending GPU-box run)
* ✅ Time-series backtesting framework applied to LSTMs — `08_time_series_backtesting.ipynb`,
  5-fold expanding window, univariate model beats persistence in 5/5 folds
* ⏳ Comprehensive evaluation documentation (pending)

**Documentation** (Completed - NEW):
* ✅ Methodology documentation (`docs/methodology.md`)
* ✅ Evaluation methodology documentation (`docs/evaluation_methodology.md`)
* ✅ Data dictionary updated with technical specifications
* ✅ Project plan updated with implementation/validation status split

**Uncertainty Quantification & Extended Horizons** (Phase 4.1 - Components 1-2 done, univariate):
* ✅ 10-model LSTM ensemble trained and evaluated — `09_lstm_ensemble.ipynb` (univariate; rerun on
  05's best variant once available)
* ✅ MC Dropout implementation with uncertainty calibration analysis — `10_mc_dropout.ipynb`
  (univariate)
* ⏳ Direct multi-horizon Seq2Seq extended to 14-day and 30-day (no autoregression) (pending)
* ⏳ Enhanced encoder-decoder with attention mechanism (pending)
* ✅ Prediction interval coverage and sharpness metrics — `picp`/`mpiw` in `evaluation_utils`;
  ensemble badly overconfident (PICP 0.17 @ 90%), MC Dropout well-covered but very wide (PICP 0.94,
  MPIW ~36x the ensemble's)
* ⏳ Error accumulation analysis for extended horizons (pending — needs Component 3)
* ✅ Uncertainty vs error correlation analysis — weak for both methods (MC Dropout 0.15, ensemble
  0.04), suggesting error here is mostly aleatoric rather than epistemic

**Spatial-Temporal CNN-LSTM** (Phase 6 - Planned):
* ⏳ Gridded ERA5 data preprocessing pipeline (Arctic Stereographic) (pending)
* ⏳ CNN spatial encoder implemented (ResNet or VGG-style) (pending)
* ⏳ LSTM temporal decoder for scalar predictions (pending)
* ⏳ Hybrid CNN-LSTM + tabular features architecture (pending)
* ⏳ Multi-horizon spatial-temporal predictions (pending)
* ⏳ Spatial interpretability analysis (GradCAM, attention maps) (pending)
* ⏳ Ablation studies (CNN vs tabular vs hybrid) (pending)
* ⏳ Computational cost documentation (pending)
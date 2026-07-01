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

**Implementation Status**: Planned

**Goal**: Extend LSTM experiments to quantify prediction uncertainty and evaluate longer forecast horizons using ensemble methods, dropout-based uncertainty, and autoregressive architectures.

**Components**:

1. **LSTM Ensemble (10 Models)**
   * Train 10 independent LSTM models with different random initializations
   * Use best-performing architecture from Phase 4 (multivariate lagged variant)
   * Generate prediction intervals from ensemble spread (mean, std, percentiles)
   * Compare ensemble mean vs single-model performance
   * Analyze inter-model variance as proxy for epistemic uncertainty
   * Document computational costs and convergence patterns across ensemble members

2. **MC Dropout for Uncertainty Estimation**
   * Implement Monte Carlo Dropout during inference (dropout enabled at test time)
   * Generate N forward passes per input (e.g., N=50 or N=100)
   * Compute prediction statistics: mean, standard deviation, confidence intervals
   * Compare MC Dropout uncertainty estimates vs ensemble uncertainty
   * Analyze uncertainty calibration (reliability diagrams)
   * Evaluate whether high-uncertainty predictions correlate with higher errors

3. **Autoregressive Predictions for Extended Horizons**
   * Implement autoregressive forecasting loop: use t+1 prediction as input for t+2
   * Extend forecast horizon beyond 7 days (test 14-day and 30-day horizons)
   * Compare autoregressive single-step LSTM vs direct Seq2Seq multi-step predictions
   * Analyze error accumulation patterns over extended horizons
   * Evaluate uncertainty growth with forecast lead time
   * Document trade-offs: flexibility (single-step) vs stability (Seq2Seq)

4. **Enhanced Encoder-Decoder LSTM**
   * Implement attention mechanism for Seq2Seq architecture
   * Add teacher forcing during training (scheduled sampling)
   * Test bidirectional encoder for improved context modeling
   * Multi-horizon outputs: 7-day, 14-day, 30-day forecast sequences
   * Compare enhanced Seq2Seq vs vanilla version from Phase 4
   * Analyze attention weights to understand temporal dependencies

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
     * Seq2Seq: predict extent for t+1 to t+7
     * Autoregressive: iteratively predict longer horizons (7-30 days)

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
* [ ] **Statistical significance** (Diebold-Mariano) — in `07_model_comparison.ipynb`
* [ ] **Multi-horizon error analysis** (Seq2Seq day 1-7) — 06 pending GPU run
* [ ] **Model comparison** (all models, identical metrics) — `07_model_comparison.ipynb`
* [ ] **Lessons documented** (what worked, what didn't, why)

**M5.1 – Uncertainty Quantification & Extended Horizons**

**Ensemble Models**:
* [ ] Train 10-model LSTM ensemble with different initializations
* [ ] Implement ensemble prediction statistics (mean, std, percentiles)
* [ ] Evaluate ensemble mean vs single-model performance
* [ ] Analyze epistemic uncertainty from ensemble spread
* [ ] Document computational costs across ensemble members

**MC Dropout**:
* [ ] Implement MC Dropout inference (50-100 forward passes)
* [ ] Generate prediction intervals from dropout sampling
* [ ] Compare MC Dropout vs ensemble uncertainty estimates
* [ ] Analyze uncertainty calibration (reliability diagrams)
* [ ] Evaluate high-uncertainty prediction correlation with errors

**Autoregressive Predictions**:
* [ ] Implement autoregressive forecasting loop (t+1 → t+2 → ... → t+N)
* [ ] Extend forecast horizon to 14-day and 30-day
* [ ] Compare autoregressive vs Seq2Seq multi-step predictions
* [ ] Analyze error accumulation over extended horizons
* [ ] Document uncertainty growth with forecast lead time

**Enhanced Encoder-Decoder**:
* [ ] Add attention mechanism to Seq2Seq LSTM
* [ ] Implement teacher forcing with scheduled sampling
* [ ] Test bidirectional encoder architecture
* [ ] Multi-horizon outputs (7-day, 14-day, 30-day)
* [ ] Analyze attention weights for temporal dependencies

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
    * Regenerates daily 1-step predictions from every checkpoint bundle in `models/`
    * **Diebold-Mariano significance** vs persistence & climatology (added to `evaluation_utils`)
    * Skill-score chart + seasonal breakdown of the best daily model
    * Laptop result: univariate LSTM beats persistence significantly (DM ≈ -9.4, p ≈ 0);
      rerun on the GPU box and the 05/06 rows fill in automatically

**Planned Notebooks** (Lower Priority)

12. **05\_ml\_baselines.ipynb** (Planned)
    * Linear regression, Ridge, Lasso, Random Forest, XGBoost
    * Multi-horizon predictions (+7, +14, +30 days)
    * Skill scores vs persistence and climatology

13. **08\_time\_series\_backtesting.ipynb** (Planned)
    * Expanding window cross-validation
    * Retrain models on multiple time periods (2019, 2020, 2021, 2022, 2023 test folds)
    * Aggregate metrics with confidence intervals
    * Statistical significance testing across folds

**Planned Notebooks** (Phase 4.1 - Uncertainty & Extended Horizons)

14. **09\_lstm\_ensemble.ipynb** (Planned)
    * Train 10 LSTM models with different random seeds
    * Ensemble prediction statistics (mean, std, percentiles)
    * Epistemic uncertainty analysis from ensemble spread
    * Comparison: ensemble mean vs best single model
    * Computational cost analysis

15. **10\_mc\_dropout.ipynb** (Planned)
    * Implement MC Dropout inference (50-100 forward passes)
    * Generate prediction intervals and uncertainty estimates
    * Reliability diagrams and calibration analysis
    * Compare MC Dropout vs ensemble uncertainty
    * Correlation between uncertainty and prediction errors

16. **11\_autoregressive\_lstm.ipynb** (Planned)
    * Autoregressive forecasting loop implementation
    * Extended horizons: 14-day and 30-day predictions
    * Error accumulation analysis over forecast lead time
    * Comparison: autoregressive vs Seq2Seq approaches
    * Uncertainty growth with horizon length

17. **12\_attention\_seq2seq.ipynb** (Planned)
    * Enhanced encoder-decoder with attention mechanism
    * Teacher forcing with scheduled sampling
    * Bidirectional encoder experiments
    * Multi-horizon outputs (7-day, 14-day, 30-day)
    * Attention weight visualization and interpretation

**Planned Notebooks** (Phase 6 - CNN-LSTM)

18. **13\_era5\_spatial\_preprocessing.ipynb** (Planned)
    * Download gridded ERA5 data (T2M, MSLP, SST, SIC)
    * Regrid to Arctic Stereographic (EPSG:3411)
    * Downsample to 64×64 and 128×128 grids
    * Create multi-channel image sequences
    * Save to zarr/HDF5 format

19. **14\_cnn\_encoder.ipynb** (Planned)
    * CNN spatial encoder implementation (ResNet/VGG-style)
    * Feature extraction from gridded weather data
    * Optional pre-training on auxiliary task
    * Spatial feature visualization

20. **15\_cnn\_lstm\_hybrid.ipynb** (Planned)
    * Full CNN-LSTM pipeline: spatial encoding + temporal modeling
    * Hybrid architecture (CNN features + tabular features)
    * Multi-horizon prediction variants
    * Training with gradient checkpointing

21. **16\_cnn\_lstm\_evaluation.ipynb** (Planned)
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
* ⏳ All models on one table with significance testing (Diebold-Mariano) — `07_model_comparison.ipynb`
* ⏳ Feature validation via ablation studies (05 variants loop provides the comparison)
* ⏳ Time-series backtesting framework applied to LSTMs (pending)
* ⏳ Comprehensive evaluation documentation (pending)

**Documentation** (Completed - NEW):
* ✅ Methodology documentation (`docs/methodology.md`)
* ✅ Evaluation methodology documentation (`docs/evaluation_methodology.md`)
* ✅ Data dictionary updated with technical specifications
* ✅ Project plan updated with implementation/validation status split

**Uncertainty Quantification & Extended Horizons** (Phase 4.1 - Planned):
* ⏳ 10-model LSTM ensemble trained and evaluated (pending)
* ⏳ MC Dropout implementation with uncertainty calibration analysis (pending)
* ⏳ Autoregressive predictions for 14-day and 30-day horizons (pending)
* ⏳ Enhanced encoder-decoder with attention mechanism (pending)
* ⏳ Prediction interval coverage and sharpness metrics (pending)
* ⏳ Error accumulation analysis for extended horizons (pending)
* ⏳ Uncertainty vs error correlation analysis (pending)

**Spatial-Temporal CNN-LSTM** (Phase 6 - Planned):
* ⏳ Gridded ERA5 data preprocessing pipeline (Arctic Stereographic) (pending)
* ⏳ CNN spatial encoder implemented (ResNet or VGG-style) (pending)
* ⏳ LSTM temporal decoder for scalar predictions (pending)
* ⏳ Hybrid CNN-LSTM + tabular features architecture (pending)
* ⏳ Multi-horizon spatial-temporal predictions (pending)
* ⏳ Spatial interpretability analysis (GradCAM, attention maps) (pending)
* ⏳ Ablation studies (CNN vs tabular vs hybrid) (pending)
* ⏳ Computational cost documentation (pending)
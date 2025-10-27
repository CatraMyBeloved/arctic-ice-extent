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
  * Pan-Arctic daily extent: `N_seaice_extent_daily_v3.0.csv`
  * Regional daily extent: `N_Sea_Ice_Index_Regional_Daily_Data_G02135_v4.0.xlsx`
  * Climatology (1981-2010): `N_seaice_extent_climatology_1981-2010_v3.0.csv`
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

2. **SARIMA models** (Completed)

   * Monthly aggregation of daily data to make SARIMA tractable
   * Model 1: SARIMA(1,0,1)×(0,1,1,12) on raw extent values
   * Model 2: SARIMA(2,0,2)×(1,0,1,12) on anomaly values
   * 40-year training period (1979-2018), 5-year test set (2019-2023)
   * Performance metrics: RMSE ~0.36-0.40 Mkm², MAE ~0.27-0.33 Mkm², MAPE ~3.5-4%

3. **Simple ML models** (Pending)

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

**Validation Status** (What has been evaluated):

* [x] Training convergence verified (all models reached early stopping)
* [x] Validation loss tracking completed (normalized MSE during training)
* [ ] **Denormalization to Mkm²** for fair comparison ← CRITICAL GAP
* [ ] **Evaluation against persistence baseline** ← CRITICAL GAP
* [ ] **Evaluation against climatology baseline** ← CRITICAL GAP
* [ ] **Statistical significance testing** of feature contributions (ablation studies)
* [ ] **Seasonal performance breakdown** (winter vs summer)
* [ ] **Multi-horizon error accumulation analysis** (Seq2Seq day 1-7)
* [ ] **Comparison to SARIMA** on identical test set with denormalized metrics
* [ ] **Comprehensive lessons documented**

**Important Notes**:

* **Normalized metrics not directly comparable**: Validation losses reported in normalized units (mean=0, std=1). Must denormalize to Mkm² before comparing to SARIMA (RMSE ~0.36 Mkm²) or baselines.
* **Features exploratory, not validated**: Lagged features and cyclical encoding added without ablation studies to confirm value. Lower validation loss observed but statistical significance not tested.
* **No baseline comparisons yet**: Cannot assess whether LSTM models beat trivial forecasts (persistence/climatology) until denormalization and standardized evaluation completed.

**Next Steps** (Phase 2-3 of Scientific Rigor Plan):

1. Implement baseline models (persistence, climatology) - `notebooks/03b_baseline_models.ipynb`
2. Create evaluation framework (`src/evaluation_utils.py`) with denormalization functions
3. Comprehensive comparison notebook (`notebooks/07_model_comparison.ipynb`) with all models on identical metrics

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
* [ ] Persistence baseline (daily data)
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
* [x] Basic LSTM architecture implemented (2-layer, 64 hidden, univariate)
* [x] Multivariate LSTM variants implemented (non-lagged, lagged, cyclical)
* [x] Seq2Seq LSTM implemented (7-day multi-horizon, vanilla encoder-decoder)
* [x] Training pipeline with early stopping, dropout, gradient clipping, LR scheduling
* [x] Models trained on 30-year dataset (1989-2019, test 2020-2023)

**Validation** ← CRITICAL GAP:
* [x] Training convergence verified (validation loss tracking)
* [ ] **Denormalization and metric standardization** (convert to Mkm²)
* [ ] **Comprehensive evaluation vs baselines** (persistence, climatology)
* [ ] **Statistical validation of feature contributions** (ablation studies, significance tests)
* [ ] **Seasonal performance breakdown** (winter vs summer)
* [ ] **Multi-horizon error analysis** (Seq2Seq day 1-7 error accumulation)
* [ ] **Model comparison** (all models on identical test set with identical metrics)
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

5. **03\_baselines.ipynb**
   * Climatology computation for all regions and variables
   * Day-of-year mean patterns
   * Visual comparison of climatology across regions

6. **03\_sarima\_baseline.ipynb**
   * SARIMA models on monthly aggregated data
   * Model 1: SARIMA on raw extent values
   * Model 2: SARIMA on anomaly values
   * Train/test split (1979-2018 / 2019-2023)
   * Performance metrics and residual diagnostics

7. **04\_basic\_lstm.ipynb** (Completed)
   * PyTorch LSTM implementation (2-layer, 64 hidden units)
   * Univariate: extent_mkm2 only
   * Custom dataset with 30-day sequence length
   * Training with early stopping, dropout, gradient clipping
   * Trained on 1989-2019, tested on 2020-2023
   * Best validation loss: ~0.002006 (normalized MSE)

8. **05\_multivariate\_lstm.ipynb** (Completed)
   * Three LSTM variants with ERA5 atmospheric variables
   * **Variant 1 - Non-lagged** (7 features): extent + t2m/msl/wind means/stds
     * Best validation loss: ~0.000323
   * **Variant 2 - Lagged** (13 features): base + extent/t2m lags (t-7, t-14, t-30)
     * Best validation loss: ~0.000306 (best overall)
   * **Variant 3 - Cyclical** (9 features): base + day_of_year sin/cos
     * Best validation loss: ~0.000321
   * Same architecture as basic LSTM (2-layer, 64 hidden, dropout 0.2)
   * Feature engineering exploration: lags, cyclical encoding
   * **Note**: Validation losses in normalized units, not yet evaluated vs baselines

9. **06\_seq2seq\_lstm.ipynb** (Completed)
   * Encoder-decoder LSTM for 7-day multi-horizon forecasting
   * 30-day input sequence → 7-day output sequence
   * Vanilla architecture (no attention, no teacher forcing)
   * Multiple variants: univariate, multivariate, cyclical
   * Best validation losses: ~0.001419-0.001631 (higher than single-step due to multi-day difficulty)
   * **Note**: Multi-horizon error accumulation analysis pending

**Experiments**

10. **experiments/shapefiles.ipynb**
    * Shapefile exploration and region definitions
    * NSIDC region visualization

**Planned Notebooks** (High Priority for Scientific Rigor)

11. **03b\_baseline\_models.ipynb** (Planned - HIGH PRIORITY)
    * Persistence baseline (y_t+1 = y_t)
    * Climatology baseline (day-of-year mean)
    * Evaluation on 2020-2023 test period
    * Seasonal breakdown (winter vs summer)
    * Establishes minimum skill thresholds for all models

12. **07\_model\_comparison.ipynb** (Planned - HIGH PRIORITY)
    * Load all trained models (SARIMA, LSTM variants, Seq2Seq)
    * **Denormalize LSTM predictions to Mkm²** (critical step)
    * Unified evaluation on identical test set (2020-2023 daily)
    * Metrics: RMSE, MAE, MAPE, skill scores vs baselines
    * Statistical significance testing (Diebold-Mariano)
    * Seasonal performance analysis (winter vs summer)
    * Multi-horizon error accumulation (Seq2Seq)
    * Export consolidated results to `results/model_comparison.csv`

**Planned Notebooks** (Lower Priority)

13. **05\_ml\_baselines.ipynb** (Planned)
    * Linear regression, Ridge, Lasso, Random Forest, XGBoost
    * Multi-horizon predictions (+7, +14, +30 days)
    * Skill scores vs persistence and climatology

14. **08\_time\_series\_backtesting.ipynb** (Planned)
    * Expanding window cross-validation
    * Retrain models on multiple time periods (2019, 2020, 2021, 2022, 2023 test folds)
    * Aggregate metrics with confidence intervals
    * Statistical significance testing across folds

**Planned Notebooks** (Phase 4.1 - Uncertainty & Extended Horizons)

15. **09\_lstm\_ensemble.ipynb** (Planned)
    * Train 10 LSTM models with different random seeds
    * Ensemble prediction statistics (mean, std, percentiles)
    * Epistemic uncertainty analysis from ensemble spread
    * Comparison: ensemble mean vs best single model
    * Computational cost analysis

16. **10\_mc\_dropout.ipynb** (Planned)
    * Implement MC Dropout inference (50-100 forward passes)
    * Generate prediction intervals and uncertainty estimates
    * Reliability diagrams and calibration analysis
    * Compare MC Dropout vs ensemble uncertainty
    * Correlation between uncertainty and prediction errors

17. **11\_autoregressive\_lstm.ipynb** (Planned)
    * Autoregressive forecasting loop implementation
    * Extended horizons: 14-day and 30-day predictions
    * Error accumulation analysis over forecast lead time
    * Comparison: autoregressive vs Seq2Seq approaches
    * Uncertainty growth with horizon length

18. **12\_attention\_seq2seq.ipynb** (Planned)
    * Enhanced encoder-decoder with attention mechanism
    * Teacher forcing with scheduled sampling
    * Bidirectional encoder experiments
    * Multi-horizon outputs (7-day, 14-day, 30-day)
    * Attention weight visualization and interpretation

**Planned Notebooks** (Phase 6 - CNN-LSTM)

19. **13\_era5\_spatial\_preprocessing.ipynb** (Planned)
    * Download gridded ERA5 data (T2M, MSLP, SST, SIC)
    * Regrid to Arctic Stereographic (EPSG:3411)
    * Downsample to 64×64 and 128×128 grids
    * Create multi-channel image sequences
    * Save to zarr/HDF5 format

20. **14\_cnn\_encoder.ipynb** (Planned)
    * CNN spatial encoder implementation (ResNet/VGG-style)
    * Feature extraction from gridded weather data
    * Optional pre-training on auxiliary task
    * Spatial feature visualization

21. **15\_cnn\_lstm\_hybrid.ipynb** (Planned)
    * Full CNN-LSTM pipeline: spatial encoding + temporal modeling
    * Hybrid architecture (CNN features + tabular features)
    * Multi-horizon prediction variants
    * Training with gradient checkpointing

22. **16\_cnn\_lstm\_evaluation.ipynb** (Planned)
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
* ⏳ Baseline models implemented (persistence, climatology) - HIGH PRIORITY
* ⏳ Evaluation framework created (`src/evaluation_utils.py`) - HIGH PRIORITY
* ⏳ LSTM predictions denormalized to Mkm² - HIGH PRIORITY
* ⏳ All models evaluated on identical test set with standardized metrics - HIGH PRIORITY
* ⏳ Statistical significance testing (model comparisons) - HIGH PRIORITY
* ⏳ Seasonal performance breakdown (winter vs summer) - HIGH PRIORITY
* ⏳ Feature validation via ablation studies (pending)
* ⏳ Time-series backtesting framework implemented (pending)
* ⏳ Multi-horizon error accumulation analysis (pending)
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
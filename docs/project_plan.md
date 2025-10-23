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

### Phase 5 – Advanced Features (Longer-Term)

* Expand to regional models (Tier B)
* Add more ERA5 variables (winds, geopotential)
* Implement ice-edge band approach (Tier C)
* Advanced spatio-temporal modeling

---

### Phase 6 – CNN-LSTM Experiment (Advanced Learning-Oriented)

* **Spatial ERA5 preprocessing pipeline**
  * Regrid ERA5 variables to Arctic Stereographic projection (EPSG:3411)
  * Downsample to computationally feasible grid (64×64 or 128×128)
  * Create multi-channel "image" sequences (T2M, SST, MSLP)
* **CNN-LSTM architecture implementation**
  * CNN spatial feature extraction from gridded weather data (→ feature vectors)
  * LSTM temporal modeling of spatial feature sequences
  * Multi-horizon prediction heads (+7, +14, +30 days)
* **Spatio-temporal evaluation**
  * Compare against aggregated-feature LSTM from Phase 4
  * Analyze CNN activation patterns and regional importance
  * Evaluate both pan-Arctic extent and regional breakdown predictions
* **Memory-efficient training**
  * Modular training of CNN and LSTM components
  * Use sliding window approach for long temporal sequences
  * Document computational trade-offs and optimization strategies

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

**M6 – Advanced Features**

* [ ] Regional models for all 14 Arctic regions
* [ ] Additional ERA5 variables (geopotential, longwave radiation)
* [ ] Cross-regional feature engineering
* [ ] Expanded evaluation across regions

**M7 – CNN-LSTM Experiment**

* [ ] ERA5 spatial preprocessing pipeline implemented
* [ ] Arctic Stereographic regridding and downsampling working
* [ ] CNN-LSTM architecture trained and evaluated
* [ ] CNN activation pattern analysis completed
* [ ] Computational optimization strategies documented

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
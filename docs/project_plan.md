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

### 3.1 NSIDC (Sea Ice Concentration, daily GeoTIFF/shapefile)

* **Derived products**:

  * Sea ice extent (≥15% SIC threshold, using area-per-pixel grid)
  * Daily climatology (1991–2020) for anomalies
* **Storage**: Raw GeoTIFFs → PostGIS (with metadata, polygons, extent timeseries)

### 3.2 ERA5 (Zarr on Google Cloud)

* **Start with 3 variables**: 2m temperature (`t2m`), sea surface temperature (`sst`), mean sea-level pressure (`msl`)
* **Features (pan-Arctic, later regional)**:

  * Mean, anomaly, 15th percentile, 85th percentile
  * Add lags (`t-1`, `t-7`, `t-30`) and rolling windows in modeling phase
* **Storage**: Aggregated features → **Parquet**

---

## 4. **System Architecture**

* **PostgreSQL + PostGIS**

  * Tables: `regions`, `climatology`, `daily_extent`, metadata
  * Use PostGIS functions for spatial aggregation and region operations

* **Parquet Feature Store**

  * Schema (initial):

    ```
    date, region, variable, stat, value
    ```
  * Partitioned by year; stored locally or cloud

* **Raw Data Access**

  * ERA5: read from Zarr via xarray/dask, cached after aggregation
  * NSIDC: local downloads, ingested into PostGIS

---

## 5. **Implementation Phases**

### Phase 0 – Warmup (Sanity Check)

* Load a single NSIDC file + single ERA5 day
* Plot both on the same projection (matplotlib/cartopy)
* Store results in PostGIS + Parquet

### Phase 1 – Data Pipeline

1. **NSIDC ingestion → PostGIS**

   * 1–2 years only
   * Compute daily pan-Arctic extent using area grid
   * Create climatology baseline (1991–2020) for anomalies

2. **ERA5 aggregation → Parquet**

   * Aggregate pan-Arctic stats (means, percentiles, anomalies)
   * Write daily records to Parquet
   * Test partitioning schemes

3. **Exploratory Analysis**

   * Plot daily time series, anomalies, climatologies
   * Check data consistency across years

---

### Phase 2 – Baseline Modeling

1. Define **targets**: extent anomaly shifted by +7 days (`y = anom.shift(-7)`)
2. Create **baselines**:

   * Persistence (`y_hat = anomaly_t`)
   * Climatology (`y_hat = mean anomaly by day-of-year`)
3. Fit simple models:

   * Linear regression
   * Ridge / Lasso
   * Random Forest, XGBoost (CPU-friendly)
4. Metrics:

   * RMSE, MAE, correlation
   * Skill scores vs persistence and climatology

---

### Phase 3 – Time-Series Specific Modeling

* Add **lags and rolling features** (t-1, t-7, t-30)
* Seasonal encoding (`sin(doy)`, `cos(doy)`)
* Validation: **expanding window backtest**
* Direct multi-horizon models: separate regressors for +7, +14, +30

---

### Phase 4 – Neural Network Experiment (Optional, Learning-Oriented)

* Small LSTM prototype (only +7d horizon, minimal variables)
* Trained on CPU; small-scale due to compute limits
* Compare with tree ensembles

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

* [x] One-day NSIDC + ERA5 joined, plotted, stored in DB/Parquet

**M1 – Basic Pipeline**

* [x] 1 year of NSIDC ingested into PostGIS
* [x] Pan-Arctic extent + anomaly time series computed
* [x] ERA5 aggregated + stored in Parquet (1 year)

**M2 – EDA**

* [ ] Time series and climatology plots
* [ ] Seasonal anomaly plots
* [ ] Validate baseline patterns

**M3 – Baselines + Simple ML**

* [ ] Persistence & climatology baselines
* [ ] Linear / Ridge / Random Forest trained on +7 horizon
* [ ] Skill scores reported

**M4 – Time-Series Models**

* [ ] Lagged features & expanding-window backtesting
* [ ] Multi-horizon models (+7, +14, +30)

**M5 – LSTM Experiment**

* [ ] Small-scale prototype trained
* [ ] Lessons documented

**M6 – Advanced Features**

* [ ] Regional breakdowns (Tier B)
* [ ] Edge-based features (Tier C)
* [ ] Expanded evaluation

**M7 – CNN-LSTM Experiment**

* [ ] ERA5 spatial preprocessing pipeline implemented
* [ ] Arctic Stereographic regridding and downsampling working
* [ ] CNN-LSTM architecture trained and evaluated
* [ ] CNN activation pattern analysis completed
* [ ] Computational optimization strategies documented

---

## 8. **Notebook Structure (Documentation Backbone)**

1. **01\_data\_ingestion\_nsidc.ipynb**

   * Load and process NSIDC data → PostGIS
   * Compute pan-Arctic extent + anomalies

2. **02\_data\_ingestion\_era5.ipynb**

   * Access ERA5 Zarr with xarray/dask
   * Aggregate daily pan-Arctic stats → Parquet

3. **03\_database\_and\_parquet\_demo.ipynb**

   * Example PostGIS spatial queries
   * Example Parquet partitioning + read performance test

4. **04\_exploratory\_analysis.ipynb**

   * Time series, climatology, seasonal cycle plots
   * Joint NSIDC/ERA5 feature exploration

5. **05\_baseline\_models.ipynb**

   * Persistence, climatology baselines
   * Linear, Ridge, Random Forest

6. **06\_time\_series\_models.ipynb**

   * Lag features, expanding window backtests
   * Horizon-specific models

7. **07\_lstm\_experiment.ipynb**

   * Minimal LSTM setup
   * Training and evaluation vs baselines

8. **08\_evaluation\_and\_summary.ipynb**

   * Consolidated metrics, seasonal skill tables
   * Model comparisons and discussion

9. **09\_future\_extensions.ipynb** *(optional)*

   * Regional models, edge bands, more variables

10. **10\_cnn\_lstm\_experiment.ipynb**

    * ERA5 spatial preprocessing and regridding pipeline
    * CNN-LSTM architecture design and implementation
    * Spatio-temporal training and validation workflows
    * CNN activation pattern analysis and regional importance mapping
    * Performance comparison with aggregated-feature models

---

## 9. **Success Criteria**

* ✅ Working PostGIS + Parquet pipeline
* ✅ Multiple models trained, compared against persistence/climatology
* ✅ Time-series backtests implemented
* ✅ LSTM prototype run (even if not strong)
* ✅ EDA and evaluations well-documented in notebooks
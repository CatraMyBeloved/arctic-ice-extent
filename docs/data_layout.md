# **Sea Ice Extent Anomaly Forecasting – Project Plan**

## 1. **Goal**

Build a system that uses NSIDC and ERA5 data to predict **Arctic sea ice extent anomaly** at multiple forecast horizons (daily, weekly, monthly), using a tiered aggregation approach and sequence models (LSTMs).

---

## 2. **Data Sources**

### 2.1 NSIDC

* **Primary**: Sea ice concentration (GeoTIFF) and shapefiles (daily, polar stereographic).
* **Derived**:

  * Extent (≥15% SIC threshold) aggregated over:

    * Pan-Arctic
    * Standard basin/sea regions
    * Edge bands (iceward/oceanward belts)
  * Climatology (1991–2020) for extent to compute anomalies.

### 2.2 ERA5 (via Google Cloud Zarr)

* **Variables** (initial list):

  * 2m temperature (`t2m`)
  * Sea surface temperature (`sst`)
  * 10m winds (`u10`, `v10`)
  * Mean sea-level pressure (`msl`)
  * Total precipitation (`tp`) *(optional initially)*
* **Derived features**:

  * Area-weighted means, percentiles, std. dev.
  * Anomalies vs. climatology
  * Rolling windows (7, 30, 90 days)
  * Lags (t−1, t−7, t−30)
  * Wind speed, wind curl, temperature–SST contrast

---

## 3. **System Architecture**

### 3.1 Storage

* **Raw data**: Public GCS buckets (ERA5 Zarr), NSIDC GeoTIFFs/shapefiles in object storage.
* **PostgreSQL + PostGIS**:

  * Stores geometry definitions for regions & edge bands
  * Metadata/versioning for spatial definitions
  * Joins spatial/temporal IDs but does not store raw rasters
* **Parquet feature store**:

  * Partitioned by `horizon`, `tier`, `region_id`, `year`
  * Contains compact, ready-to-train daily records

### 3.2 Tiers of Aggregation

1. **Tier A** – Pan-Arctic (single region)
2. **Tier B** – Basins/seas (multi-region)
3. **Tier C** – Edge bands (dynamic per date; optional per region)

---

## 4. **Processing Pipeline**

### Step 1 – Region definitions (PostGIS)

* Load polygons for:

  * Entire Arctic (Tier A)
  * Basins/seas (Tier B)
  * Ocean mask for edge band clipping
* Store as `regions` table with valid date ranges.

### Step 2 – Edge detection (for Tier C)

* From daily NSIDC concentration:

  * Extract ice edge (15% contour)
  * Compute signed distance transform
  * Create inner/outer 250 km bands
* Store in `edge_bands` table (geometry per date, side, region).

### Step 3 – ERA5 aggregation

* Open ERA5 Zarr lazily via xarray/dask in cloud.
* Spatially subset to Arctic region.
* Aggregate ERA5 vars over each geometry (Tier A, B, C).
* Compute stats, anomalies, lags, and rolling windows.

### Step 4 – Feature store creation

* Write daily features per `(date, region_id, tier, horizon)` to Parquet.
* Partition for fast queries and training.
* Include target variables (`sie_anom_target`, `sie_raw`, `sie_clim`).

---

## 5. **Model Training**

### 5.1 Input shapes

* **Tier A only**: `(batch, seq_len, n_features_A)`
* **A + B**: Concatenate or pool region features before LSTM
* **A + B + C**: Same as above with additional pseudo-regions for edge bands

### 5.2 Forecast horizons

* Daily (+1), weekly (+7), monthly (+30)
* Multi-head output or separate models

### 5.3 Baselines

* Persistence (`y(t+H) = y(t)`)
* Seasonal-naïve (climatology for day-of-year)

---

## 6. **Milestones**

1. **M1 – Setup infrastructure**

   * Postgres/PostGIS database
   * GCS bucket for project data
   * Parquet storage layout agreed

2. **M2 – Tier A pipeline**

   * Aggregate NSIDC → extent anomaly
   * ERA5 → daily features over Arctic
   * Join into Parquet table
   * Train baseline + LSTM for D+7

3. **M3 – Tier B pipeline**

   * Add basin/sea regions
   * Aggregate ERA5 + extent anomaly per basin
   * Retrain and evaluate vs. Tier A

4. **M4 – Tier C pipeline**

   * Implement edge detection + bands
   * Aggregate ERA5 features in bands
   * Retrain and evaluate vs. previous tiers

5. **M5 – Refinement**

   * Feature engineering improvements
   * Hyperparameter tuning for LSTM
   * Robustness checks by season/decade

---

## 7. **Directory Layout Proposal**

```
/data
    /nsidc/raw
    /era5/raw
    /features
        horizon=D7
            tier=A
                region_id=ARCTIC_ALL
                    year=1995
                    ...
            tier=B
            tier=C
/models
/notebooks
/sql
/docs
```


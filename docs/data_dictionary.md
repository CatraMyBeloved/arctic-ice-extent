# Sea-Ice Project — Data Dictionary

## Scope & Naming

* Naming: `nh_` (Northern), `sh_` (Southern), `regional_` for basin tables; suffixes `_daily`, `_monthly`, `_doy`.
* Units: areas in **km²**; rates in **km²/day** or **Mkm²/month** (state in column name).
* Time zone: dates are UTC calendar days.
* Provenance: every fact table carries `source` and `ingested_at`.

---

## Core fact tables

### nh\_extent\_daily

* **Grain:** 1 row per **date** (pan-Arctic)
* **Primary key:** `(date)`
* **Indexes:** `(date)`
* **Description:** Daily sea-ice **extent** (≥15% SIC), pan-Arctic.

| Column        | Type        | Unit | Description                                |
|---------------|-------------|------|--------------------------------------------|
| date          | date        | —    | UTC date                                   |
| extent\_km2   | numeric     | km²  | Total pan-Arctic extent                    |
| missing\_flag | smallint    | 0/1  | Source gap indicator (if present)          |
| source        | text        | —    | Upstream dataset tag (e.g., “G02135 v4.0”) |
| ingested\_at  | timestamptz | —    | Ingest timestamp                           |

---

### nh\_extent\_monthly

* **Grain:** 1 row per **year, month** (pan-Arctic)
* **Primary key:** `(year, month)`
* **Indexes:** `(year, month)`
* **Description:** Monthly pan-Arctic extent (from NSIDC monthly or aggregated).

| Column       | Type        | Unit | Description              |
|--------------|-------------|------|--------------------------|
| year         | integer     | —    | Calendar year            |
| month        | integer     | 1–12 | Calendar month           |
| extent\_km2  | numeric     | km²  | Monthly extent           |
| anomaly\_km2 | numeric     | km²  | Optional, vs. 1981–2010  |
| stddev\_km2  | numeric     | km²  | Optional monthly std dev |
| source       | text        | —    | Upstream dataset tag     |
| ingested\_at | timestamptz | —    | Ingest timestamp         |

---

### nh\_regional\_extent\_daily

* **Grain:** 1 row per **date, region**
* **Primary key:** `(date, region_id)`
* **Indexes:** `(region_id, date)`
* **Description:** Daily extent per standard Arctic region.

| Column       | Type        | Unit | Description                 |
|--------------|-------------|------|-----------------------------|
| date         | date        | —    | UTC date                    |
| region\_id   | text        | —    | FK → `regions.region_id`    |
| extent\_km2  | numeric     | km²  | Extent (≥15% SIC) in region |
| source       | text        | —    | Upstream dataset tag        |
| ingested\_at | timestamptz | —    | Ingest timestamp            |

---

### nh\_regional\_area\_daily

* **Grain:** 1 row per **date, region**
* **Primary key:** `(date, region_id)`
* **Indexes:** `(region_id, date)`
* **Description:** Daily **area** per region (concentration-weighted).

| Column       | Type        | Unit | Description          |
|--------------|-------------|------|----------------------|
| date         | date        | —    | UTC date             |
| region\_id   | text        | —    | FK → `regions`       |
| area\_km2    | numeric     | km²  | Ice area in region   |
| source       | text        | —    | Upstream dataset tag |
| ingested\_at | timestamptz | —    | Ingest timestamp     |

---

### nh\_regional\_extent\_monthly / nh\_regional\_area\_monthly

* **Grain:** 1 row per **year, month, region**
* **Primary key:** `(year, month, region_id)`
* **Indexes:** `(region_id, year, month)`
* **Description:** Monthly extent/area per region.

| Column                  | Type        | Unit | Description    |
|-------------------------|-------------|------|----------------|
| year                    | integer     | —    | Year           |
| month                   | integer     | 1–12 | Month          |
| region\_id              | text        | —    | FK → `regions` |
| extent\_km2 / area\_km2 | numeric     | km²  | Monthly value  |
| anomaly\_km2            | numeric     | km²  | Optional       |
| source                  | text        | —    | Upstream tag   |
| ingested\_at            | timestamptz | —    | Ingest ts      |

---

### nh\_extent\_climatology\_doy

* **Grain:** 1 row per **day-of-year** (1981–2010 base)
* **Primary key:** `(doy)`
* **Indexes:** `(doy)`
* **Description:** DOY climatology used for anomalies.

| Column            | Type        | Unit  | Description             |
|-------------------|-------------|-------|-------------------------|
| doy               | integer     | 1–366 | Day of year             |
| mean\_extent\_km2 | numeric     | km²   | Mean extent (1981–2010) |
| std\_extent\_km2  | numeric     | km²   | Std dev (1981–2010)     |
| source            | text        | —     | Climatology source      |
| ingested\_at      | timestamptz | —     | Ingest ts               |

---

### nh\_rates\_of\_change\_daily

* **Grain:** 1 row per **date** (pan-Arctic)
* **Primary key:** `(date)`
* **Indexes:** `(date)`
* **Description:** Daily change metrics derived from extent.

| Column                      | Type        | Unit    | Description          |
|-----------------------------|-------------|---------|----------------------|
| date                        | date        | —       | UTC date             |
| delta\_km2\_per\_day        | numeric     | km²/day | Day-over-day change  |
| roll5\_delta\_km2\_per\_day | numeric     | km²/day | 5-day smoothed       |
| source                      | text        | —       | Derivation reference |
| ingested\_at                | timestamptz | —       | Ingest ts            |

*(Monthly variant: `nh_rates_of_change_monthly` with `delta_mkm2_per_month`.)*

---

### nh\_min\_max\_rankings

* **Grain:** 1 row per ranked event
* **Primary key:** `(rank_type, rank, hemisphere)`
* **Indexes:** `(hemisphere, rank_type, rank)`
* **Description:** Historical minima/maxima rankings.

| Column       | Type        | Unit | Description                |
|--------------|-------------|------|----------------------------|
| hemisphere   | text        | —    | `NH`/`SH`                  |
| rank\_type   | text        | —    | `daily`, `5day`, `monthly` |
| rank         | integer     | —    | 1 = extreme                |
| date         | date        | —    | Date of event              |
| extent\_km2  | numeric     | km²  | Value                      |
| notes        | text        | —    | Tie/metadata               |
| source       | text        | —    | Upstream tag               |
| ingested\_at | timestamptz | —    | Ingest ts                  |

---

## Reference & metadata tables

### regions

* **Grain:** 1 row per region definition (versioned)
* **Primary key:** `(region_id, valid_from)`
* **Indexes:** `USING GIST (geom)`, `(region_id)`, `(valid_from, valid_to)`
* **Description:** Canonical region polygons.

| Column      | Type     | Unit | Description                   |
|-------------|----------|------|-------------------------------|
| region\_id  | text     | —    | Stable ID (e.g., `BEAUFORT`)  |
| name        | text     | —    | Human-readable                |
| hemisphere  | text     | —    | `NH`/`SH`                     |
| geom        | geometry | —    | PostGIS polygon/multipolygon  |
| valid\_from | date     | —    | Start validity                |
| valid\_to   | date     | —    | End validity (NULL = current) |
| source      | text     | —    | Origin of geometry            |
| crs\_epsg   | integer  | —    | e.g., 3411                    |
| area\_km2   | numeric  | km²  | Optional pre-computed         |

---

### calendar

* **Grain:** 1 row per date
* **Primary key:** `(date)`
* **Indexes:** `(year)`, `(month)`, `(doy)`
* **Description:** Date lookup for joins.

| Column   | Type    | Unit  | Description       |
|----------|---------|-------|-------------------|
| date     | date    | —     | UTC date          |
| year     | integer | —     | Calendar year     |
| month    | integer | 1–12  | Calendar month    |
| doy      | integer | 1–366 | Day of year       |
| season   | text    | —     | `DJF/MAM/JJA/SON` |
| is\_leap | boolean | —     | Leap day flag     |

---

### dataset\_versions

* **Grain:** 1 row per ingest artifact
* **Primary key:** `(dataset_id, ingested_at)`
* **Indexes:** `(dataset_id)`, `(checksum)`
* **Description:** Provenance and audit trail.

| Column        | Type        | Unit | Description                                 |
|---------------|-------------|------|---------------------------------------------|
| dataset\_id   | text        | —    | Logical name (e.g., `G02135_daily_geotiff`) |
| version       | text        | —    | Upstream version (e.g., `v4.0`)             |
| source\_path  | text        | —    | URL or bucket path                          |
| checksum      | text        | —    | File/bundle hash                            |
| record\_count | integer     | —    | Rows ingested                               |
| notes         | text        | —    | Free text                                   |
| ingested\_at  | timestamptz | —    | Ingest ts                                   |

---

### vector\_catalog  *(for shapefiles like mean extent DOY)*

* **Grain:** 1 row per vector asset
* **Primary key:** `(asset_id)`
* **Indexes:** `(product, doy)`, `(crs_epsg)`
* **Description:** Registry of vector files.

| Column      | Type    | Unit  | Description                            |
|-------------|---------|-------|----------------------------------------|
| asset\_id   | text    | —     | Unique ID                              |
| product     | text    | —     | e.g., `extent_mean_doy`, `median_edge` |
| path        | text    | —     | Storage path/URL                       |
| crs\_epsg   | integer | —     | e.g., 3411                             |
| doy         | integer | 1–366 | For DOY assets                         |
| month       | integer | 1–12  | For monthly assets                     |
| version     | text    | —     | Upstream version                       |
| valid\_from | date    | —     | Start validity                         |
| valid\_to   | date    | —     | End validity                           |

---

### raster\_catalog  *(for daily GeoTIFFs: concentration, extent)*

* **Grain:** 1 row per raster asset
* **Primary key:** `(asset_id)`
* **Indexes:** `(product, date)`, `(crs_epsg)`
* **Description:** Registry of raster files.

| Column                  | Type        | Unit | Description                |
|-------------------------|-------------|------|----------------------------|
| asset\_id               | text        | —    | Unique ID                  |
| product                 | text        | —    | `concentration` / `extent` |
| date                    | date        | —    | Acquisition date           |
| path                    | text        | —    | Storage path/URL           |
| crs\_epsg               | integer     | —    | e.g., 3411                 |
| spatial\_resolution\_km | numeric     | km   | Nominal pixel size         |
| version                 | text        | —    | Upstream version           |
| checksum                | text        | —    | Optional                   |
| ingested\_at            | timestamptz | —    | Ingest ts                  |

---

## Suggested views (for convenience)

* **`nh_extent_anomaly_daily_v`**
  `nh_extent_daily` joined to `calendar` (for `doy`) and `nh_extent_climatology_doy` to expose `anomaly_km2` as a column.

* **`nh_regional_extent_anomaly_daily_v`**
  Same as above but per `region_id` (if you maintain a regional climatology later).

---

## QA & constraints

* CHECK: non-negative areas/extent (`extent_km2 >= 0`, `area_km2 >= 0`).
* FK: `region_id` in regional tables → `regions.region_id`.
* Consistent CRS: `regions.crs_epsg` must match rasters/shapes referenced for aggregation (document in catalogs).


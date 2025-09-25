# Sea-Ice Project — Data Dictionary

## Current Database Schema (As Implemented)

**Database Connection:** `postgresql://postgres:OgadJKGiH@localhost:5432/seaice`

The following tables currently exist in the database:

### ice\_extent\_pan\_arctic\_daily
* **Grain:** 1 row per **date, region**
* **Primary key:** `(region, date)`
* **Indexes:** `UNIQUE INDEX ice_extent_pan_arctic_daily_pk (region, date)`
* **Description:** Pan-Arctic daily sea-ice extent data (appears to support regional breakdown)

| Column        | Type               | Nullable | Description                     |
|---------------|--------------------|-----------|---------------------------------|
| date          | date               | NO       | UTC date                        |
| region        | character varying  | NO       | Region identifier               |
| extent_mkm2   | double precision   | YES      | Ice extent in million km² (Mkm²)|

### ice\_extent\_regional\_daily
* **Grain:** 1 row per **date, region**
* **Primary key:** `(region, date)`
* **Indexes:** `UNIQUE INDEX ice_extent_regional_daily_pk (region, date)`
* **Description:** Daily sea-ice extent per region

| Column | Type               | Nullable | Description       |
|--------|--------------------|----------|-------------------|
| date   | date               | NO       | UTC date          |
| extent | double precision   | YES      | Ice extent (units unclear) |
| region | character varying  | NO       | Region identifier |

### ice\_extent\_climatology
* **Grain:** 1 row per **day-of-year**
* **Primary key:** None defined
* **Description:** Climatological statistics by day of year with percentile data

| Column    | Type             | Nullable | Description              |
|-----------|------------------|----------|--------------------------|
| dayofyear | bigint           | YES      | Day of year (1-366)      |
| avg_extent| double precision | YES      | Average extent           |
| std_dev   | double precision | YES      | Standard deviation       |
| p10       | double precision | YES      | 10th percentile          |
| p25       | double precision | YES      | 25th percentile          |
| p50       | double precision | YES      | 50th percentile (median) |
| p75       | double precision | YES      | 75th percentile          |
| p90       | double precision | YES      | 90th percentile          |

### PostGIS System Tables
* **geography_columns** (VIEW): PostGIS geography metadata
* **geometry_columns** (VIEW): PostGIS geometry metadata
* **spatial_ref_sys** (BASE TABLE): Spatial reference systems

---

## Data Conventions

* **Units:** Ice extent in million km² (Mkm²) as stored in `extent_mkm2` columns
* **Time zone:** All dates are UTC calendar days
* **Naming:** Uses descriptive table names without prefixes
* **Indexes:** All data tables use compound primary keys on `(region, date)` for efficient time-series queries
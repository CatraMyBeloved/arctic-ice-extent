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

| Column      | Type               | Nullable | Description                     |
|-------------|--------------------|-----------|---------------------------------|
| date        | date               | NO       | UTC date                        |
| extent_mkm2 | double precision   | YES      | Ice extent in million km² (Mkm²)|
| region      | character varying  | NO       | Region identifier               |

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

---

## Parquet Feature Store Schema

### File Organization

**Location**: `data/processed/parquet/`

**File naming**: `era5_regional_{year}.parquet`
- Example: `era5_regional_2020.parquet`
- Partitioned by year for efficient access
- Each file contains all regions and variables for that year

### Schema Specification

**Format**: Long-format (tidy data)

| Column    | Type     | Description |
|-----------|----------|-------------|
| date      | date     | UTC calendar date |
| region    | string   | Region identifier (matches NSIDC-0780 naming) |
| variable  | string   | ERA5 variable name (t2m, msl, u10, v10, tp, wind_speed) |
| stat_type | string   | Statistic type (mean, std, p15, p85) |
| value     | float64  | Statistic value in converted units |

**Example rows**:
```
date       | region      | variable | stat_type | value
-----------|-------------|----------|-----------|--------
2020-01-01 | pan_arctic  | t2m      | mean      | -15.3
2020-01-01 | pan_arctic  | t2m      | std       | 8.2
2020-01-01 | pan_arctic  | msl      | mean      | 1013.2
2020-01-01 | beaufort    | t2m      | mean      | -22.1
```

**Rationale**:
- Long format enables flexible querying (filter by variable, region, stat type)
- Parquet columnar format optimizes for analytical queries
- Year partitioning enables efficient date range queries

### Query Patterns

**Load all variables for one region**:
```python
df = pd.read_parquet('data/processed/parquet/era5_regional_2020.parquet')
pan_arctic_data = df[df['region'] == 'pan_arctic']
```

**Load specific variable across all regions**:
```python
df = pd.read_parquet('data/processed/parquet/era5_regional_2020.parquet')
temp_data = df[df['variable'] == 't2m']
```

**Load multiple years**:
```python
dfs = []
for year in range(2018, 2024):
    dfs.append(pd.read_parquet(f'data/processed/parquet/era5_regional_{year}.parquet'))
combined = pd.concat(dfs, ignore_index=True)
```

---

## Unit Conversions and Variable Definitions

### ERA5 Variable Transformations

All unit conversions applied during ERA5 data transformation pipeline (notebook `01c_data_ingestion_era5_transformation.ipynb`):

| Variable       | Source Name | Source Unit        | Target Unit | Conversion Formula |
|----------------|-------------|--------------------|--------------|--------------------|
| Temperature    | t2m         | Kelvin (K)         | Celsius (°C) | `°C = K - 273.15` |
| Pressure       | msl         | Pascal (Pa)        | Hectopascal (hPa) | `hPa = Pa / 100` |
| U-wind         | u10         | meters/second (m/s)| m/s          | No conversion |
| V-wind         | v10         | meters/second (m/s)| m/s          | No conversion |
| Precipitation  | tp          | meters (cumulative)| meters       | No conversion |
| Wind Speed     | (derived)   | —                  | m/s          | `sqrt(u10² + v10²)` |

### Derived Variables

**Wind Speed Calculation**:
```python
wind_speed = np.sqrt(u10**2 + v10**2)
```
- **Rationale**: Total wind magnitude from orthogonal components
- **Physical interpretation**: Surface wind speed (10 meters above surface)
- **Use case**: Wind stress on sea ice surface

**Statistical Aggregations**:

For each variable and region, the following statistics are computed:
- **mean**: Arithmetic mean across all grid cells in region
- **std**: Standard deviation across grid cells (captures spatial variability)
- **p15**: 15th percentile (lower bound of typical values)
- **p85**: 85th percentile (upper bound of typical values)

**Example interpretation** (t2m for Beaufort Sea):
- `mean = -22.1°C`: Average temperature across Beaufort region
- `std = 5.3°C`: Spatial temperature variability within region
- `p15 = -26.8°C`: Colder areas of region
- `p85 = -17.5°C`: Warmer areas of region

---

## Data Validation Rules

### Ice Extent Constraints

**Physical constraints**:
```python
# Ice extent must be non-negative
assert extent_mkm2 >= 0, "Negative ice extent physically impossible"

# Ice extent maximum bounded by region area
# Pan-Arctic: ~20 Mkm² approximate maximum
# Individual regions: Varies by region size
```

**Temporal constraints**:
```python
# Valid date range
assert date >= '1979-01-01', "NSICD data starts 1979"
assert date <= datetime.today(), "Cannot have future data"
```

**Completeness checks**:
- Check for unexpected missing values (NaN) in recent years (post-1987)
- 1979-1989 expected to have interpolated values (every-other-day original data)

### ERA5 Variable Ranges

**Reasonable value ranges** (Pan-Arctic annual):

| Variable | Typical Min | Typical Max | Notes |
|----------|-------------|-------------|-------|
| t2m      | -40°C       | +20°C       | Arctic temperatures |
| msl      | 950 hPa     | 1050 hPa    | Sea level pressure |
| u10      | -30 m/s     | +30 m/s     | U-wind component |
| v10      | -30 m/s     | +30 m/s     | V-wind component |
| wind_speed | 0 m/s     | +40 m/s     | Total wind speed |
| tp       | 0 m        | 0.01 m/day  | Daily precipitation |

**Outlier detection**: Values outside typical ranges should be flagged for review (possible data quality issues or extreme weather events)

### Region Name Validation

**Valid region names** (must match NSICD-0780 v1.0 shapefile):

```python
VALID_REGIONS = [
    'pan_arctic',
    'beaufort',
    'chukchi',
    'east_siberian',
    'laptev',
    'kara',
    'barents',
    'greenland',
    'baffin',
    'canadian_archipelago',
    'hudson',
    'central_arctic',
    'bering',
    'baltic',
    'okhotsk'
]
```

**Note**: Region names use lowercase with underscores (snake_case convention)

---

## Climatology Specification

### Baseline Period

**Current implementation**: All available years (1979-2023) used to compute climatology
**WMO standard**: 1981-2010 (30-year normal period)

**Inconsistency note**: Project documentation originally specified 1981-2010, but implementation uses full period. Future decision needed on which to adopt.

### Computation Method

**Day-of-year aggregation**:
```python
climatology = data.groupby(data['date'].dt.dayofyear).agg({
    'extent_mkm2': ['mean', 'std',
                    lambda x: x.quantile(0.10),  # p10
                    lambda x: x.quantile(0.25),  # p25
                    lambda x: x.quantile(0.50),  # p50
                    lambda x: x.quantile(0.75),  # p75
                    lambda x: x.quantile(0.90)]  # p90
})
```

**Key features**:
- 366 climatology values (one per calendar day including Feb 29)
- No smoothing window applied (raw day-of-year statistics)
- Percentiles provide distribution information beyond mean
- Stored in `ice_extent_climatology` PostgreSQL table

### Usage

**Anomaly calculation**:
```python
anomaly_mkm2 = observed_extent_mkm2 - climatology_mean[day_of_year]
```

**Climatology baseline forecasts**:
```python
forecast_extent = climatology_mean[day_of_year_of_forecast_date]
```

---

## Data Quality Notes

### Known Data Issues

**1979-1989 NSICD Interpolation**:
- Original data: Every-other-day observations
- Applied: Linear interpolation to fill missing days
- **Implication**: Interpolated values smoother than actual observations
- **Recommendation**: Be cautious with 1979-1989 high-frequency variability analysis

**ERA5 Grid Resolution**:
- Native resolution: ~31 km (0.25° × 0.25° grid)
- Regional aggregation: Simple mean across grid cells (no area weighting)
- **Implication**: Smaller regions may have fewer grid cells, less stable statistics
- **Recommendation**: Use mean for large regions; interpret std/percentiles carefully for small regions

### Completeness Status

| Data Source | Coverage | Completeness |
|-------------|----------|--------------|
| NSICD Ice Extent | 1979-01-01 to 2023-12-31 | Complete (with 1979-1989 interpolation) |
| ERA5 (processed) | 1979-01-01 to 2023-12-31 | Complete (processed and stored in Parquet) |
| Climatology | Day-of-year (1-366) | Complete |

**Last updated**: 2025 (project active development)

---

## Access Patterns

### Database Access

**Via SQLAlchemy**:
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://user:pass@localhost:5432/seaice')

# Load ice extent data
query = """
    SELECT date, region, extent_mkm2
    FROM ice_extent_pan_arctic_daily
    WHERE region = 'pan_arctic'
      AND date BETWEEN '2020-01-01' AND '2023-12-31'
    ORDER BY date
"""
df = pd.read_sql(query, engine)
```

**Via utility function**:
```python
from src.data_utils import load_data

# Load with automatic Parquet merge
df = load_data(regions='pan_arctic', years=range(2020, 2024))
```

### Parquet Access

**Via pandas**:
```python
import pandas as pd

# Single year
df = pd.read_parquet('data/processed/parquet/era5_regional_2020.parquet')

# Filter during read (efficient)
df = pd.read_parquet(
    'data/processed/parquet/era5_regional_2020.parquet',
    filters=[('region', '==', 'pan_arctic')]
)
```

**Via pyarrow** (for large datasets):
```python
import pyarrow.parquet as pq

# Read with column selection
table = pq.read_table(
    'data/processed/parquet/era5_regional_2020.parquet',
    columns=['date', 'region', 'variable', 'value']
)
df = table.to_pandas()
```

---

## Related Documentation

- `docs/methodology.md`: Data processing pipeline details, regional aggregation methodology
- `docs/evaluation_methodology.md`: Metrics and evaluation standards
- `docs/project_plan.md`: Project phases, data sources, and architecture overview
# Arctic Sea Ice Extent Forecasting

A learning-focused data science project exploring geospatial-temporal forecasting of Arctic sea ice extent using NSIDC observations and ERA5 climate reanalysis data.

## Overview

This project builds a complete pipeline for Arctic sea ice extent anomaly prediction, combining daily satellite observations with atmospheric reanalysis data. The primary goal is educational—understanding modern geospatial data workflows, spatial databases, and time series modeling—rather than production performance.

**Current Status**: Data pipeline complete (Phase 1-2), exploratory analysis in progress, baseline modeling upcoming.

## Key Features

- **Multi-source data integration**: NSIDC sea ice index + ERA5 atmospheric variables (temperature, pressure, wind, precipitation)
- **PostGIS spatial database**: Regional aggregation with proper coordinate reference systems (EPSG:3411 Arctic Stereographic)
- **Parquet feature store**: Efficient columnar storage with year-based partitioning for 45 years of daily data
- **Regional analysis**: Pan-Arctic and 14+ sub-region breakdowns (Beaufort, Barents, Laptev, etc.)
- **Learning-oriented**: Systematic progression from baselines → classical ML → neural networks (LSTM, CNN-LSTM)

## Tech Stack

**Data Processing**: Python, xarray, dask, geopandas, regionmask
**Database**: PostgreSQL + PostGIS for spatial operations
**Storage**: Parquet (pyarrow) for feature storage
**ML (planned)**: scikit-learn, PyTorch for sequence models
**Data Access**: CDS API for ERA5 reanalysis data

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL with PostGIS extension
- CDS API account (free registration at https://cds.climate.copernicus.eu/)
- ~50GB disk space for raw data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CatraMyBeloved/arctic-ice-extent.git
cd arctic-ice-extent
```

2. Set up Python environment:
```bash
python setup_env.py
```

This creates a virtual environment and installs dependencies from `requirements.in`.

3. Configure CDS API credentials:

Create `~/.cdsapirc` with your credentials:
```
url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
```

4. Configure database connection:

Edit notebook connection strings or create environment variables:
```python
DATABASE_URL = 'postgresql://postgres:password@localhost:5432/seaice'
```

5. Run data ingestion notebooks:
- `notebooks/01a_data_ingestion_era5_download.ipynb` - Download ERA5 via CDS API
- `notebooks/01b_data_ingestion_nsidc.ipynb` - Load NSIDC sea ice data into PostGIS
- `notebooks/01c_data_ingestion_era5_transformation.ipynb` - Process and aggregate ERA5 to regions

## Project Structure

```
├── data/
│   ├── raw/              # Downloaded NSIDC tables, shapefiles, ERA5 NetCDF
│   ├── interim/          # Processed monthly ERA5 files
│   └── processed/        # Parquet feature store
├── notebooks/            # Jupyter notebooks for each pipeline stage
├── src/                  # Reusable utility functions
│   ├── data_utils.py     # Data loading and merging
│   ├── coordinate_utils.py  # Geospatial transformations
│   └── plot_utils.py     # Visualization helpers
├── utilities/            # Scripts for data downloads
└── docs/                 # Project planning and documentation
```

## Learning Objectives

This project systematically explores:
- Geospatial-temporal data handling with xarray and PostGIS
- Working with climate reanalysis data via CDS API
- Database schema design for time series
- Regional spatial aggregation and masking
- Time series methodology (backtesting, anomaly detection)
- Neural network sequence modeling (LSTM architectures)

## Roadmap

- [x] **Phase 0-1**: Data pipeline (NSIDC + ERA5 ingestion)
- [x] **Phase 2**: Exploratory data analysis
- [ ] **Phase 3**: Baseline models (persistence, climatology, linear regression)
- [ ] **Phase 4**: Time series models with lagged features
- [ ] **Phase 5**: LSTM experimentation
- [ ] **Phase 6**: Advanced CNN-LSTM spatial-temporal modeling

See [docs/project_plan.md](docs/project_plan.md) for detailed milestones.

## Data Sources

- **NSIDC Sea Ice Index (G02135)**: Daily sea ice extent derived from passive microwave satellite imagery
- **ERA5 Reanalysis**: Global atmospheric data from ECMWF, accessed via CDS API
- **NSIDC Arctic Regions Shapefile**: Standard regional definitions for Arctic seas

## Documentation

- [Project Plan](docs/project_plan.md) - Detailed phase breakdown and learning goals
- [Data Dictionary](docs/data_dictionary.md) - Database schema and conventions
- [Project Story](docs/project_story.md) - Background and motivation

## License

This is an educational project. Data sources have their own licenses (NSIDC, ECMWF).

## Contact

Ole Stein
[GitHub](https://github.com/CatraMyBeloved) • [Email](mailto:ole.stein.ctr@outlook.com)
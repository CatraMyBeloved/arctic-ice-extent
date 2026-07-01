# Arctic Sea Ice Extent Forecasting

A learning-focused data science project exploring geospatial-temporal forecasting of Arctic sea ice extent using NSIDC observations and ERA5 climate reanalysis data.

## Overview

This project builds a complete pipeline for Arctic sea ice extent anomaly prediction, combining daily satellite observations with atmospheric reanalysis data. The primary goal is educational—understanding modern geospatial data workflows, spatial databases, and time series modeling—rather than production performance.

**Current Status**: Data pipeline and EDA complete. SARIMA and LSTM models (univariate, multivariate, seq2seq) implemented and trained; evaluation framework (`src/evaluation_utils.py`) built. Next up: running the persistence/climatology baselines and a standardized cross-model comparison (denormalized to Mkm²).

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
- [uv](https://docs.astral.sh/uv/) (manages the Python version and dependencies)
- PostgreSQL with PostGIS extension
- CDS API account (free registration at https://cds.climate.copernicus.eu/)
- ~50GB disk space for raw data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CatraMyBeloved/arctic-ice-extent.git
cd arctic-ice-extent
```

2. Set up the environment:
```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `pyproject.toml` (Python version is pinned in `.python-version` and fetched automatically by uv). Run commands in the environment with `uv run`, e.g. `uv run jupyter lab`.

3. Configure CDS API credentials:

Create `~/.cdsapirc` with your Personal Access Token (from your CDS profile page):
```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

4. Start the PostgreSQL + PostGIS database:
```bash
podman compose up -d    # or: docker compose up -d
```

This launches a `postgis/postgis` container matching the connection string used in the code
(`postgresql://postgres:password@localhost:5432/seaice`), with data persisted in a named volume.
See [docs/database.md](docs/database.md) for details, lifecycle commands, and the one-time
Podman socket setup. The ingestion notebooks create the `ice_extent_*` tables on first run.

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
│   ├── evaluation_utils.py  # Metrics, baseline models, backtesting, comparison
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
- [x] **Phase 3**: Baseline models (persistence, climatology, linear regression)
- [x] **Phase 4**: Time series models with lagged features
- [ ] **Phase 5**: LSTM experimentation (**IN PROGRESS**)
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

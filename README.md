# Arctic Sea Ice Extent Forecasting

A learning-focused data science project exploring geospatial-temporal forecasting of Arctic sea ice extent using NSIDC observations and ERA5 climate reanalysis data.

## Overview

This project builds a complete pipeline for Arctic sea ice extent anomaly prediction, combining daily satellite observations with atmospheric reanalysis data. The primary goal is educational—understanding modern geospatial data workflows, spatial databases, and time series modeling—rather than production performance.

**Current Status**: Data pipeline and EDA complete. Baselines, SARIMA, and the univariate LSTM (plus its 10-seed ensemble, MC Dropout, and 5-fold expanding-window backtest) trained and evaluated with statistical significance testing (Diebold-Mariano + Holm-Bonferroni) in `07_model_comparison.ipynb` — the univariate LSTM significantly beats persistence at daily scale. Multivariate (05) and seq2seq (06) notebooks are refactored and ready, pending the ERA5 parquet store and a GPU-box run. Next up: uncertainty quantification on the multivariate variant, and direct multi-horizon (14/30-day) Seq2Seq extension.

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
- [ ] **Phase 3**: Baselines (persistence, climatology, SARIMA done; simple ML — Ridge/RF/XGBoost — pending)
- [ ] **Phase 4/4.1**: LSTM experiments (**IN PROGRESS** — univariate model, its 10-seed ensemble, MC Dropout, and 5-fold backtest done; multivariate/seq2seq pending GPU box; extended horizons + skill-decay curve next)
- [ ] **Phase 6**: Spatial CNN-LSTM (stretch)

See [docs/project_plan.md](docs/project_plan.md) for the prioritized roadmap (core path vs stretch) and [docs/results_log.md](docs/results_log.md) for findings so far.

## Data Sources

- **NSIDC Sea Ice Index (G02135)**: Daily sea ice extent derived from passive microwave satellite imagery
- **ERA5 Reanalysis**: Global atmospheric data from ECMWF, accessed via CDS API
- **NSIDC Arctic Regions Shapefile**: Standard regional definitions for Arctic seas

## Documentation

- [Project Plan](docs/project_plan.md) - Goals, status overview, prioritized roadmap
- [Results Log](docs/results_log.md) - Findings per experiment, negative results, bugs caught, lessons
- [Experiment Designs](docs/experiment_designs.md) - Scoped specs for planned experiments (E1-E8)
- [Methodology](docs/methodology.md) - Data processing, architectures, split strategy
- [Evaluation Methodology](docs/evaluation_methodology.md) - Metrics, baselines, backtesting protocol
- [Data Dictionary](docs/data_dictionary.md) - Database schema and conventions
- [Project Story](docs/project_story.md) - Background and motivation

## License

This is an educational project. Data sources have their own licenses (NSIDC, ECMWF).

## Contact

Ole Stein
[GitHub](https://github.com/CatraMyBeloved) • [Email](mailto:ole.stein.ctr@outlook.com)

# Arctic Sea Ice Extent Forecasting — Project Plan

A **learning-focused** project forecasting Arctic sea ice extent (anomalies) from NSIDC
observations and ERA5 reanalysis. Performance is secondary; the goal is to understand the tools,
data, and methods properly, and to document the journey honestly — including the failures.

**Document map** — this file is the forward-looking plan only:

| Document | What lives there |
|---|---|
| this file | goals, status overview, prioritized roadmap |
| [`results_log.md`](results_log.md) | what each experiment **found** — results, negatives, bugs caught, lessons |
| [`experiment_designs.md`](experiment_designs.md) | detailed specs for planned experiments (E1–E8), each with explicit scope boundaries |
| [`methodology.md`](methodology.md) | data processing, feature engineering, model architectures, split strategy |
| [`evaluation_methodology.md`](evaluation_methodology.md) | metrics, baselines, denormalization, backtesting protocol |
| [`data_dictionary.md`](data_dictionary.md) | database schema, Parquet schema, units, conventions |
| [`database.md`](database.md) | PostGIS container setup and lifecycle |

---

## Goals

Build a complete pipeline for Arctic sea ice extent forecasting, learning along the way:

1. Geospatial-temporal data handling in Python (xarray, dask, regionmask)
2. A GIS-capable database (PostgreSQL/PostGIS) and Parquet feature storage
3. EDA for geospatial time series
4. Proper forecasting methodology: baselines, temporal splits, backtesting, significance testing
5. Classical models (persistence, climatology, SARIMA, linear/tree ensembles) before neural ones
6. LSTM sequence models, uncertainty quantification, multi-horizon forecasting
7. (Later) spatial deep learning on gridded data — zarr, CNN encoders, interpretability

*Secondary*: a basic understanding of Arctic seasonal cycles and anomaly patterns.

This is a portfolio/learning project, not a paper — the bar is "methodologically honest and
well-documented", not "novel".

## Data & infrastructure (summary)

* **NSIDC Sea Ice Index (G02135 v4.0)** — daily pan-Arctic + 14-region extent and a 1981–2010
  climatology, ingested into **PostgreSQL/PostGIS** tables.
* **ERA5 reanalysis** — t2m, msl, u10/v10, tp (1979–2023) via CDS API, aggregated to NSIDC
  regions (mean/std/p15/p85), stored as yearly long-format **Parquet** files.
* Access via `src/data_utils.py` (`load_extent_daily()` — DB only; `load_data()` — DB + ERA5
  join). Extent data auto-bootstraps via `src/data_bootstrap.ensure_extent_data()`.

Details: `data_dictionary.md` (schemas, units), `methodology.md` (processing), `database.md`
(container). The ERA5 download is ~2,700 CDS API requests — never re-trigger casually (see
`CLAUDE.md`).

## Phase overview

| Phase | Scope | Status |
|---|---|---|
| 0–1 | Data pipeline: NSIDC → PostgreSQL, ERA5 → Parquet | ✅ Done |
| 2 | EDA: seasonal cycles, correlations, trends | ✅ Done |
| 3 | Baselines: persistence, climatology, SARIMA, simple ML | ⏳ Simple ML pending (**E1**) |
| 4 | LSTM experiments on the shared engine (`src/lstm_utils.py`) | ⏳ Univariate done & evaluated; multivariate/seq2seq pending GPU run (**E2**) |
| 4.1 | Uncertainty quantification & extended horizons | ⏳ Ensemble + MC Dropout done (univariate); horizons (**E3**) and objectives (**E4**) planned |
| 5 | Regional models, more ERA5 variables | 🅿 Parking lot (**E7**) — undesigned until a gating reason appears |
| 6 | Spatial CNN-LSTM on gridded ERA5 | 🅿 Stretch (**E8**) |

Completed-phase findings live in `results_log.md`, not here.

## Notebook index

Status values: Completed / Refactored — pending GPU run / Planned (keep this convention).

| Notebook | Status | One-liner |
|---|---|---|
| `01a_data_ingestion_era5_download` | Completed | ERA5 download via CDS API (slow, external — don't re-run casually) |
| `01b_data_ingestion_nsidc` | Completed | NSIDC CSV/Excel → PostgreSQL |
| `01c_data_ingestion_era5_transformation` | Completed | ERA5 NetCDF → regional Parquet |
| `02_EDA` | Completed | Seasonal cycles, correlations, trends |
| `03a_sarima_baseline` | Completed | SARIMA on monthly data, walk-forward |
| `03b_baseline_models` | Completed | Persistence + climatology, daily |
| `03c_ml_baselines` | Planned (**E1**) | Linear/Ridge/Lasso/RF/XGBoost |
| `04_basic_lstm` | Completed | Univariate LSTM, evaluated vs baselines |
| `05_multivariate_lstm` | Pending GPU run (**E2**) | ERA5 climate-feature variants |
| `06_seq2seq_lstm` | Pending GPU run (**E2**) | 7-day direct multi-step |
| `07_model_comparison` | Completed (self-updating) | Ranked table + DM significance from `results/model_comparison.csv` |
| `08_time_series_backtesting` | Completed | 5-fold expanding window, univariate |
| `09_lstm_ensemble` | Completed (univariate) | 10-seed ensemble + intervals; rerun on best 05 variant in **E2** |
| `10_mc_dropout` | Completed (univariate) | MC Dropout intervals; rerun on best 05 variant in **E2** |
| `11_extended_horizon_seq2seq` | Planned (**E3**) | 14/30-day direct models, skill-decay curve |
| `12_attention_seq2seq` | Planned, stretch (**E5**) | Attention, teacher forcing, bidirectional encoder vs vanilla |
| `13_predictive_vae` | Planned, stretch (**E6**) | Variational seq2seq, three-way calibration comparison |
| `14`–`17` (spatial CNN-LSTM) | Planned, stretch (**E8**) | Gridded ERA5 → CNN encoder → LSTM |

## Roadmap

Full specs with scope boundaries: `experiment_designs.md`. Order within the core path matters.

### Core path (the finish line)

1. [ ] **E1 — Simple ML baselines** (`03c`): fill the gap between trivial baselines and neural
   nets; does the LSTM beat Ridge/XGBoost?
2. [ ] **E2 — GPU-box run**: 05/06 + 07 rerun + ensemble and MC Dropout reruns on the best
   variant; answers whether climate features move the 1-day noise floor.
3. [ ] **E3 — Extended horizons** (`11`): dedicated 14/30-day direct models → the
   **skill-decay curve** and zero-skill crossover. The project's headline figure.
4. [ ] **E4 — Objective-function experiments**: Step-0 extractability gate, then the gated
   subset of {differenced target, rate-of-change losses, distribution losses}; DILATE/soft-DTW
   as a follow-on if A/B show promise.
5. [ ] **Write-up**: distill the narrative into the README + `results_log.md`; skill-decay
   figure as the headline. This unchecked box is the highest-portfolio-value item in the plan.

**The project is "done" when the core path is done.** Everything below is optional depth.

### Stretch (in rough priority order, only after the core path)

* [ ] **E6 — Predictive VAE** (`13`): distribution-of-futures intervals vs ensemble/MC Dropout
  — the best-motivated stretch item given the aleatoric-error finding
* [ ] **E5 — Enhanced encoder-decoder** (`12`): attention, teacher forcing, bidirectional
  encoder; multi-horizon outputs and attention-weight analysis
* [ ] **E8 — Spatial CNN-LSTM** (`14`–`17`): gridded ERA5 → CNN encoder → LSTM, hybrid with
  tabular features, ablations and spatial interpretability — the big learning payoff and the
  real GPU workload
* [ ] **E7 — Advanced features**: regional models for all 14 regions, more ERA5 variables,
  ice-edge band approach; detailed design deferred until it moves onto the active roadmap

## Evaluation strategy

Unchanged and non-negotiable (details in `evaluation_methodology.md`):

* Every model vs persistence **and** climatology, denormalized to Mkm², on identical test eras.
* Three-way temporal split — validation era for all model selection, test era touched once.
* Diebold-Mariano significance (Holm-Bonferroni across families); expanding-window backtesting
  for the headline claims.
* One comparison log: `results/model_comparison.csv` via `log_model_results()`, tagged by
  `scale`; comparisons only within a scale.

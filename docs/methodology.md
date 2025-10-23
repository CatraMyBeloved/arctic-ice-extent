# Methodology Documentation

## Overview

This document details the data processing, feature engineering, and modeling approaches used in the Arctic Sea Ice Extent forecasting project. All design choices are documented as first-iteration exploratory implementations prioritizing learning over optimization.

---

## Data Processing Methodology

### Temporal Coverage and Interpolation

**NSIDC Sea Ice Extent Data (1979-present)**
- **Source**: NSIDC G02135
- **Temporal resolution**:
  - Daily from 1987-present
  - Every-other-day from 1979-1989
- **Interpolation strategy**: Linear interpolation applied to fill missing days in 1979-1989 period
- **Quality assurance**: NSIDC handles quality control internally with strict boundaries; no additional filtering applied in this project
- **Units**: Million km² (Mkm²), stored as `extent_mkm2` in database

**ERA5 Climate Reanalysis (1979-present)**
- **Source**: Copernicus Climate Data Store (CDS) via API
- **Variables selected**:
  - `t2m`: 2-meter air temperature (Kelvin)
  - `msl`: Mean sea level pressure (Pascal)
  - `u10`, `v10`: 10-meter wind components (m/s)
  - `tp`: Total precipitation (meters, cumulative)
- **Temporal resolution**: Daily mean values aggregated from hourly data
- **Download method**: CDS API with programmatic retrieval via `cdsapi` Python package

### Coordinate Systems and Spatial Processing

**Coordinate Reference System**
- All spatial processing performed in **EPSG:4326** (WGS84 lat/lon)
- NSICD spatial gridded data not currently used (regional extent values used directly from NSIDC tables)
- ERA5 native projection: EPSG:4326 (no reprojection needed)
- **Note**: Future phases may incorporate NSIDC gridded data in EPSG:3411 (NSIDC Sea Ice Polar Stereographic North) for spatial modeling

**Regional Definitions**
- Regions defined by **NSIDC-0780 v1.0** shapefiles
- **Total regions**: 15 (14 Arctic regions + Pan-Arctic aggregate)
- **Regions list**: Pan-Arctic, Beaufort Sea, Chukchi Sea, East Siberian Sea, Laptev Sea, Kara Sea, Barents Sea, Greenland Sea, Baffin Bay, Canadian Archipelago, Hudson Bay, Central Arctic, Bering Sea, Baltic Sea, Sea of Okhotsk

### Regional Aggregation Methodology

**ERA5 Grid-to-Region Aggregation**

The project aggregates gridded ERA5 data to regional statistics using the following approach:

1. **Masking approach**: Point-in-polygon geometric test using region shapefiles
   - Each ERA5 grid cell centroid tested against region boundary polygons
   - Grid cells fully within region boundary are included

2. **Aggregation method**: Simple arithmetic mean (no area weighting)
   - All included grid cells weighted equally regardless of cell area
   - Statistics computed: mean, standard deviation, 15th percentile, 85th percentile

3. **Cell inclusion**: All grid cells within region boundary included (land and ocean cells)
   - No land/ocean masking applied
   - Rationale: Atmospheric variables over land still influence sea ice dynamics

4. **Rationale**: Capture large-scale atmospheric patterns and circulation features rather than precise local values
   - Regional means smooth local variability
   - Focus on synoptic-scale atmospheric forcing
   - Acceptable for learning-focused project; production systems may require area-weighted aggregation

**Ice Extent Regional Aggregation**
- NSIDC provides regional extent values directly in tabular format
- No spatial aggregation needed
- Values ingested directly into PostgreSQL database
- Units: Million km² (Mkm²) stored as `extent_mkm2` column

### Unit Conversions

All unit conversions applied during ERA5 data transformation pipeline:

| Variable          | Source Unit       | Target Unit | Conversion Formula            |
|-------------------|-------------------|-------------|-------------------------------|
| Temperature (t2m) | Kelvin (K)        | Celsius (°C)| `°C = K - 273.15`             |
| Pressure (msl)    | Pascal (Pa)       | Hectopascal (hPa) | `hPa = Pa / 100`        |
| Wind Speed        | u10, v10 (m/s)    | wind_speed (m/s) | `wind_speed = sqrt(u10² + v10²)` |
| Precipitation (tp)| meters (cumulative)| meters      | No conversion (stored as-is)  |

**Notes**:
- Wind speed derived from u and v components using Pythagorean theorem
- Total precipitation stored in meters (cumulative daily) as provided by ERA5
- Temperature conversion ensures intuitive interpretation (°C more familiar than K)
- Pressure conversion to hPa aligns with meteorological conventions

---

## Feature Engineering

### Temporal Features

**Lagged Variables**

Exploratory lag choices implemented for capturing temporal dependencies:

- **t-7**: 7-day lag
- **t-14**: 14-day lag
- **t-30**: 30-day lag

**Variables lagged**:
- `extent_mkm2` (ice extent)
- `t2m_mean` (temperature)

**Rationale**:
- These lags were chosen exploratorily to test their influence on prediction power
- **Not optimized**: Selection not based on autocorrelation function analysis or domain-specific sea ice dynamics timescales
- **First iteration**: Represents exploratory feature engineering subject to future refinement
- **Future work**: Systematic lag selection via ACF analysis, domain expert consultation, or automated feature selection

**Validation status**: Lagged features implemented but not yet rigorously validated via ablation studies (see `docs/evaluation_methodology.md` for planned validation approach)

**Cyclical Encoding**

Day-of-year encoded as sine/cosine pair to capture annual seasonality:

```python
day_of_year_sin = sin(2π × day_of_year / 365.25)
day_of_year_cos = cos(2π × day_of_year / 365.25)
```

**Rationale**:
- Preserves continuity between December 31 (day 365) and January 1 (day 1)
- Avoids artificial discontinuity that linear day-of-year encoding would create
- Two-dimensional encoding (sin/cos pair) enables neural networks to learn circular patterns
- 365.25 denominator accounts for leap years

**Other cyclical features**: None currently implemented (month-of-year, hour-of-day not needed for daily aggregated data)

**Validation status**: Implemented but impact not yet rigorously assessed

### Climatology and Anomaly Features

**Climatology Baseline**

- **Computation method**: Simple mean across all years for each day-of-year
  - 366 climatology values (one per calendar day including Feb 29)
  - Computed independently for each region and variable

- **Baseline period**: All available years used to compute climatology
  - **Current implementation**: 1979-2023 (all data)
  - **WMO standard**: 1981-2010 (30-year normal period)
  - **Inconsistency note**: Project plan originally stated 1981-2010; implementation uses all years
  - **Future decision needed**: Adopt WMO 1981-2010 standard or document rationale for using full period

- **Smoothing**: No smoothing window applied
  - Raw day-of-year means without moving average
  - Retains high-frequency variability in climatological baseline

- **Storage**: Climatology values stored in PostgreSQL `ice_extent_climatology` table with columns: `dayofyear`, `avg_extent`, `std_dev`, `p10`, `p25`, `p50`, `p75`, `p90`

**Anomaly Calculation**

Anomalies computed as deviation from climatological mean:

```python
anomaly_mkm2 = observed_extent_mkm2 - climatology_mean_extent_mkm2
```

**Interpretation**:
- Positive anomaly: Above-average ice extent for that calendar day
- Negative anomaly: Below-average ice extent for that calendar day
- Removes strong seasonal component, revealing interannual variability and trends

**Use cases**:
- SARIMA Model 2: Trained on anomaly values instead of raw extent
- Feature engineering: Anomaly values can be additional model inputs alongside raw values
- Trend analysis: Long-term trends more visible in anomaly time series

---

## Model Architectures and Design Choices

### SARIMA Baseline Models

**Model Selection Process**

1. **Stationarity assessment**: Augmented Dickey-Fuller (ADF) test applied to raw and anomaly time series
   - Both series found non-stationary (p > 0.05)
   - Differencing required for ARIMA modeling

2. **Order selection approach**: Combination of grid search and manual inspection
   - ACF/PACF plots inspected for AR and MA order selection
   - Multiple candidate models fit
   - AIC/BIC criterion used to compare candidate models
   - Residual diagnostics (ACF of residuals, normality tests) validated final choice

3. **Seasonal period**: s=12 (monthly data)
   - Monthly aggregation necessary for computational tractability
   - Daily SARIMA with s=365 prohibitively expensive (~16,000 observations)
   - Monthly aggregation (~540 observations) preserves seasonal patterns while reducing computation

**Final Model Specifications**

**Model 1: SARIMA on Raw Ice Extent Values**
- **Order**: SARIMA(1, 0, 1) × (0, 1, 1, 12)
- **Non-seasonal components**:
  - AR(1): First-order autoregressive term
  - I(0): No non-seasonal differencing applied
  - MA(1): First-order moving average term
- **Seasonal components**:
  - SAR(0): No seasonal autoregressive term
  - SI(1): First-order seasonal differencing (removes annual cycle)
  - SMA(1): First-order seasonal moving average
  - s=12: Monthly seasonal period
- **Training**: 1979-2018 (480 months)
- **Test**: 2019-2023 (60 months)
- **Performance**: RMSE=0.3572 Mkm², MAE=0.2668 Mkm², MAPE=3.53%

**Model 2: SARIMA on Anomaly Values**
- **Order**: SARIMA(2, 0, 2) × (1, 0, 1, 12)
- **Strategy**: Model anomalies (deviations from climatology) instead of raw values
- **Rationale**:
  - Anomalies remove strong seasonal component
  - May be more stationary (less differencing needed: d=0, D=0)
  - Can improve forecast accuracy for interannual variability
- **Non-seasonal components**:
  - AR(2): Second-order autoregressive term
  - I(0): No non-seasonal differencing (anomalies more stationary)
  - MA(2): Second-order moving average term
- **Seasonal components**:
  - SAR(1): First-order seasonal autoregressive term
  - SI(0): No seasonal differencing needed (seasonality removed via anomaly calculation)
  - SMA(1): First-order seasonal moving average
  - s=12: Monthly seasonal period
- **Training**: Same as Model 1 (1979-2018)
- **Test**: Same as Model 1 (2019-2023)
- **Performance**: RMSE=0.4056 Mkm², MAE=0.3259 Mkm², MAPE=4.01%
- **Conversion**: Anomaly forecasts converted back to raw extent via: `predicted_extent = predicted_anomaly + climatology_mean`

**Rationale for Monthly Aggregation**:
- Daily SARIMA with s=365 computationally prohibitive
- Monthly aggregation smooths day-to-day weather noise while maintaining seasonal patterns
- Enables longer historical baseline (1979 start vs 1989 for LSTM due to memory constraints)
- Trade-off: Loses high-frequency variability but gains temporal coverage

---

### LSTM Architectures

All LSTM variants share common architectural choices with differences only in input features.

**Basic LSTM (Univariate)**

- **Architecture**: 2-layer stacked LSTM
- **Hidden units**: 64 per layer
- **Lookback window**: 30 days (sequence length)
- **Input features**: `extent_mkm2` only (univariate)
- **Output**: Single value (extent at t+1)
- **Dropout**: 0.2 between LSTM layers (regularization)
- **Batch size**: 32
- **Optimizer**: Adam with learning rate 0.001
- **Learning rate scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
  - Reduces learning rate when validation loss plateaus
- **Loss function**: MSE (Mean Squared Error)
- **Early stopping**: Patience=15 epochs
  - Stops training if validation loss doesn't improve for 15 consecutive epochs
- **Gradient clipping**: Max norm 1.0 (prevents exploding gradients)
- **Training data**: 1989-2019 (11,322 daily samples after sequence creation)
- **Test data**: 2020-2023 (1,461 daily samples)
- **Normalization**: Z-score normalization per feature using training set mean/std
  - All input features scaled to mean=0, std=1
  - **Critical**: Predictions must be denormalized for interpretation and comparison

**Hyperparameter Rationale**:
- **30-day lookback**: Chosen due to computational limitations on CPU
  - Planned future work: Test longer windows (60, 90 days) to capture monthly/seasonal memory
  - Trade-off: Longer lookback = more temporal context but higher memory and computation
- **64 hidden units**: Exploratory choice based on initial experiments
  - Range 60-80 units showed reasonable performance in preliminary tests
  - **Not optimized**: Lacks rigorous hyperparameter tuning (grid search, Bayesian optimization)
  - Production systems would require systematic tuning
- **Learning rate 0.001**: Standard Adam default
  - No learning rate search performed
  - Future work: Learning rate scheduling from start, warmup periods
- **2 layers**: Balance between model capacity and overfitting risk
  - Deeper networks (3-4 layers) not explored due to small dataset concerns

**Design Philosophy**: Prioritize simplicity and interpretability over performance for learning purposes

---

**Multivariate LSTM (Three Variants)**

All multivariate variants use identical LSTM architecture (2-layer, 64 hidden units, same hyperparameters as basic LSTM). Only input features differ.

**Variant 1: Non-Lagged Features (7 features)**

**Feature list**:
1. `extent_mkm2` - Ice extent (target variable also used as input)
2. `t2m_mean` - Temperature mean
3. `t2m_std` - Temperature standard deviation
4. `msl_mean` - Sea level pressure mean
5. `msl_std` - Sea level pressure standard deviation
6. `wind_speed_mean` - Wind speed mean
7. `wind_speed_std` - Wind speed standard deviation

**Rationale**:
- Include both mean and standard deviation to capture regional variability
- Atmospheric variables known to influence sea ice (temperature, pressure systems, wind stress)
- Current-day values only (no temporal lags)

**Training samples**: 11,292 (after 30-day sequence creation)
**Test samples**: 1,431
**Best validation loss**: 0.000323 (normalized MSE)

**Variant 2: With Lagged Features (13 features)**

**Feature list**: All base features (7) plus lagged features (6):

Base features (7):
- `extent_mkm2`, `t2m_mean`, `t2m_std`, `msl_mean`, `msl_std`, `wind_speed_mean`, `wind_speed_std`

Lagged features (6):
8. `extent_mkm2_lag7` - Ice extent 7 days ago
9. `extent_mkm2_lag14` - Ice extent 14 days ago
10. `extent_mkm2_lag30` - Ice extent 30 days ago
11. `t2m_mean_lag7` - Temperature 7 days ago
12. `t2m_mean_lag14` - Temperature 14 days ago
13. `t2m_mean_lag30` - Temperature 30 days ago

**Rationale**:
- Capture recent trends in ice extent and temperature
- Lags at weekly, bi-weekly, and monthly intervals
- Extent lags: Persistence/momentum in ice dynamics
- Temperature lags: Lagged thermal forcing effects

**Training samples**: 11,262 (fewer due to 30-day lag requiring more initial data dropped)
**Test samples**: 1,401
**Best validation loss**: 0.000306 (normalized MSE) - slight improvement over non-lagged

**Validation status**: Improvement observed but not statistically validated. Ablation studies needed to confirm individual lag contributions.

**Variant 3: With Cyclical Encoding (9 features)**

**Feature list**: All base features (7) plus cyclical time encoding (2):

Base features (7):
- `extent_mkm2`, `t2m_mean`, `t2m_std`, `msl_mean`, `msl_std`, `wind_speed_mean`, `wind_speed_std`

Cyclical features (2):
8. `day_of_year_sin` - sin(2π × day_of_year / 365.25)
9. `day_of_year_cos` - cos(2π × day_of_year / 365.25)

**Rationale**:
- Explicit seasonal signal provided to network
- Arctic sea ice has strong seasonal cycle (maximum March, minimum September)
- Test whether explicit seasonality encoding helps vs LSTM learning seasonality from data

**Training samples**: 11,292
**Test samples**: 1,431
**Best validation loss**: 0.000321 (normalized MSE) - comparable to non-lagged, slightly worse than lagged

**Observation**: Cyclical encoding did not improve performance substantially. LSTM may already capture seasonality from temporal sequences. Further investigation needed.

---

**Architecture Consistency**

**Rationale for uniform architecture**: Isolate impact of feature engineering choices
- Same LSTM architecture (2-layer, 64 hidden) across all variants
- Same hyperparameters (lr, dropout, early stopping)
- Same train/test split (1989-2019 / 2020-2023)
- Only input features vary
- **Goal**: Determine which features matter most without confounding architectural differences

**Limitation**: Optimal architecture may differ for different feature sets (e.g., more features may benefit from wider networks). Future work: Joint hyperparameter and feature selection.

---

### Seq2Seq LSTM (Sequence-to-Sequence)

**Architecture Type**: Vanilla encoder-decoder (no attention mechanism, no teacher forcing)

**Encoder**:
- 2-layer LSTM (64 hidden units per layer)
- Input sequence: 30 days historical data
- Processes full input sequence
- Final hidden state (h_n, c_n) passed to decoder

**Decoder**:
- 2-layer LSTM (64 hidden units per layer)
- Output sequence: 7 days forecast
- Initialized with encoder's final hidden state
- Generates predictions autoregressively (no teacher forcing)
  - Day 1 prediction fed as input for day 2
  - Day 2 prediction fed as input for day 3
  - ... continuing for 7 days

**Multi-Horizon Forecasting**:
- **Input**: 30-day sequence (days t-29 to t)
- **Output**: 7-day sequence (days t+1 to t+7)
- **Use case**: Direct multi-day forecast without iterative single-step predictions

**Variants**:
- **Univariate**: `extent_mkm2` only
- **Multivariate**: Same feature sets as multivariate LSTM variants (non-lagged, lagged, cyclical)

**Training Configuration**:
- Same hyperparameters as basic LSTM (lr=0.001, early stopping patience=15)
- Same train/test split (1989-2019 / 2020-2023)
- Batch size: 32
- Loss: MSE averaged over all 7 forecast days

**Performance**:
- **Univariate Seq2Seq**: ~0.001419 validation loss (normalized MSE)
- **Multivariate Seq2Seq**: ~0.001631 validation loss
- **Note**: Higher loss than single-step models expected (multi-day forecasting harder than 1-day ahead)

**Limitations (Current Implementation)**:
- **No attention mechanism**: Decoder cannot focus on relevant encoder time steps
  - Entire input sequence compressed into fixed-size hidden state
  - May lose information for longer sequences
- **No teacher forcing**: During training, decoder uses its own predictions as inputs
  - More realistic (matches inference mode) but harder to train
  - Teacher forcing (using true values during training) often accelerates learning

**Future Enhancements Planned**:
- Attention mechanism: Allow decoder to dynamically focus on relevant input time steps
- Teacher forcing: Stabilize training by providing true values during training phase
- Longer forecast horizons: Extend to 14-day, 30-day predictions

**Validation Status**: Architecture implemented and trained. Comprehensive evaluation pending (error accumulation analysis, comparison to iterated single-step forecasts, seasonal performance).

---

## Train/Test Split Strategy

### LSTM Models (Daily Data)

**Training Period**: 1989-2019
- **Duration**: 30 years (11,322 daily observations)
- **Start date**: 1989-01-01 (limited by ERA5 data processing; could extend to 1979 with full pipeline)
- **End date**: 2019-12-31

**Test Period**: 2020-2023
- **Duration**: 4 years (1,461 daily observations)
- **Start date**: 2020-01-01
- **End date**: 2023-12-31

**Rationale**:
- Preserves temporal order (critical for time series)
- 30-year training period provides sufficient data for neural networks
- 4-year test period covers recent anomalous years (low ice extent)
- No validation set explicitly held out (validation loss computed on test set during training - **limitation noted**)

**Proper validation approach** (future work):
- Split training into train (1989-2014) and validation (2015-2019)
- Reserve 2020-2023 as true held-out test set
- Prevents test set leakage via early stopping on test performance

**Sample Sizes** (after 30-day sequence creation):
- Basic LSTM: 11,292 training samples, 1,431 test samples
- Multivariate (lagged): 11,262 training samples, 1,401 test samples (fewer due to lag dropping initial rows)

---

### SARIMA Models (Monthly Data)

**Training Period**: 1979-2018
- **Duration**: 40 years (480 months)
- **Start date**: 1979-01
- **End date**: 2018-12

**Test Period**: 2019-2023
- **Duration**: 5 years (60 months)
- **Start date**: 2019-01
- **End date**: 2023-12

**Rationale**:
- Longer historical baseline enabled by monthly aggregation
- Full NSIDC data range utilized (1979 start)
- Monthly aggregation computationally necessary for SARIMA
- 40-year training captures multiple climate regimes and variability modes

**Comparison Note**: SARIMA and LSTM use different time periods
- SARIMA: 1979-2018 (40 years monthly)
- LSTM: 1989-2019 (30 years daily)
- **Limitation**: Not identical training data, but test periods overlap (2020-2023 vs 2019-2023)
- For fair comparison: Evaluate both on 2020-2023 test set with consistent temporal alignment

---

### Data Normalization

**LSTM Models**: Z-score normalization applied
```python
normalized_value = (value - training_mean) / training_std
```

**Normalization strategy**:
- Compute mean and std from training set only (no data leakage)
- Apply same normalization parameters to test set
- Per-feature normalization (each feature scaled independently)
- Enables stable gradient descent and balanced feature contributions

**Denormalization (Critical)**:
```python
denormalized_value = (normalized_value × training_std) + training_mean
```

**Requirement**: All LSTM predictions must be denormalized before:
- Computing evaluation metrics (RMSE, MAE, MAPE)
- Comparing to SARIMA models (which use raw Mkm² values)
- Visualizing predictions
- Reporting results

**Status**: Denormalization not yet applied in existing notebooks. See `docs/evaluation_methodology.md` for standardized denormalization protocol.

**SARIMA Models**: No normalization applied (operates on raw Mkm² values or anomaly values)

---

## Critical Limitations and Future Work

### Current Limitations

1. **Single train/test split**: No cross-validation or rolling window backtesting
   - High risk: Single split may be lucky/unlucky
   - **Solution**: Implement expanding window cross-validation (see `docs/evaluation_methodology.md`)

2. **Hyperparameters not optimized**: All LSTM hyperparameters exploratory
   - No grid search, random search, or Bayesian optimization performed
   - **Solution**: Systematic hyperparameter tuning for production systems

3. **Feature selection not validated**: Lagged features and cyclical encoding added without ablation studies
   - Unclear which features actually contribute value
   - **Solution**: Ablation studies removing features one-at-a-time

4. **No baseline comparisons yet**: LSTM models not compared to persistence or climatology baselines
   - Cannot assess whether models beat naïve forecasts
   - **Solution**: Implement baselines and compute skill scores (Phase 2 of plan)

5. **Normalized vs raw metrics**: LSTM validation losses not comparable to SARIMA RMSE
   - Validation loss (0.000306) vs RMSE (0.36 Mkm²) are different scales
   - **Solution**: Denormalize all LSTM predictions (Phase 3 of plan)

6. **Test set used for early stopping**: Validation loss computed on test set (data leakage)
   - Proper approach: Separate validation set for early stopping
   - **Solution**: Implement proper train/validation/test split

7. **No statistical significance testing**: Model differences not tested for significance
   - Unclear if performance differences meaningful or due to chance
   - **Solution**: Diebold-Mariano tests, paired t-tests (Phase 3 of plan)

### Future Enhancements

1. **Attention mechanisms**: Implement attention for Seq2Seq models
2. **Teacher forcing**: Stabilize Seq2Seq training
3. **Longer lookback windows**: Test 60, 90-day sequences (if computationally feasible)
4. **Regional models**: Expand from pan-Arctic to individual Arctic regions
5. **Additional variables**: Incorporate sea surface temperature, geopotential height, radiation variables
6. **Ensemble methods**: Combine predictions from multiple models
7. **Uncertainty quantification**: Prediction intervals, ensemble spread
8. **Transfer learning**: Pre-train on reanalysis data, fine-tune on observations

---

## Learning Project Context

This project prioritizes **educational value and methodological exploration** over production-grade performance optimization:

- **Feature choices**: Exploratory implementations to learn feature engineering workflows
- **Hyperparameters**: Initial guesses to understand impact, not optimized for best performance
- **Model architectures**: Pedagogically clear implementations (standard LSTM, vanilla Seq2Seq)
- **Evaluation framework**: Evolving as part of learning process (Phase 2-3 of project plan)

**Goal**: Develop hands-on understanding of geospatial-temporal data science workflows, time series forecasting methodologies, and neural network development practices.

All design decisions represent **first iterations** intended to facilitate learning. Rigorous optimization and production deployment would require systematic refinement following evaluation methodology documented in `docs/evaluation_methodology.md`.

---

## References and Resources

**Data Sources**:
- NSIDC G02135: https://nsidc.org/data/g02135
- NSIDC-0780 Shapefiles: https://nsidc.org/data/nsidc-0780
- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
- Copernicus Climate Data Store: https://cds.climate.copernicus.eu/

**Methodological References**:
- SARIMA: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control.
- LSTM: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
- Seq2Seq: Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks.

**Related Project Documentation**:
- `docs/evaluation_methodology.md`: Standardized evaluation framework and metrics
- `docs/data_dictionary.md`: Database schema and data specifications
- `docs/project_plan.md`: Project phases, milestones, and implementation status

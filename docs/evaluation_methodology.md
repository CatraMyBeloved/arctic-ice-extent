# Evaluation Methodology

## Overview

This document establishes standardized evaluation protocols for assessing forecast model performance in the Arctic Sea Ice Extent project. All models must be evaluated using consistent metrics, test sets, and comparison frameworks to enable fair, scientifically rigorous conclusions.

**Core Principle**: A model is only as good as its evaluation. Implementation without validation is speculation.

---

## Evaluation Metrics

### Standard Forecast Error Metrics

**Root Mean Squared Error (RMSE)**

```
RMSE = sqrt(mean((y_true - y_pred)²))
```

**Interpretation**:
- Units: Same as predicted variable (Mkm² for ice extent)
- Range: [0, ∞), lower is better
- Penalizes large errors more heavily than small errors (due to squaring)
- **Use when**: Large errors are particularly costly or undesirable
- **Example**: RMSE = 0.35 Mkm² means typical forecast error is ~0.35 million km²

**Mean Absolute Error (MAE)**

```
MAE = mean(|y_true - y_pred|)
```

**Interpretation**:
- Units: Same as predicted variable (Mkm²)
- Range: [0, ∞), lower is better
- Treats all errors equally (no squaring)
- More robust to outliers than RMSE
- **Use when**: Want interpretable "average error magnitude"
- **Example**: MAE = 0.25 Mkm² means average absolute forecast error is 0.25 million km²

**Mean Absolute Percentage Error (MAPE)**

```
MAPE = mean(|y_true - y_pred| / |y_true|) × 100%
```

**Interpretation**:
- Units: Percentage (%)
- Range: [0, ∞), lower is better
- Scale-independent: Enables comparison across different magnitude variables
- **Use when**: Want relative error perspective or comparing models across different scales
- **Limitation**: Undefined when y_true = 0, biased toward under-prediction
- **Example**: MAPE = 3.5% means forecasts off by 3.5% on average

---

### Skill Scores (Relative Performance)

**Skill Score vs Baseline**

```
Skill_Score = 1 - (RMSE_model / RMSE_baseline)
```

**Interpretation**:
- Range: (-∞, 1]
  - SS = 1: Perfect forecast (RMSE = 0)
  - SS = 0: No better than baseline
  - SS > 0: Model beats baseline
  - SS < 0: Model worse than baseline
- Unitless: Enables comparison across different variables and scales
- **Use when**: Need to demonstrate model provides value over simple alternatives
- **Critical**: All models should be evaluated against persistence and climatology baselines

**Example**:
- Persistence baseline: RMSE = 0.50 Mkm²
- Your LSTM model: RMSE = 0.35 Mkm²
- Skill score: 1 - (0.35 / 0.50) = 0.30 (30% improvement over persistence)

**Minimum acceptance criterion**: SS > 0 (beat baseline)
**Good performance**: SS > 0.20 (20%+ improvement)
**Excellent performance**: SS > 0.40 (40%+ improvement)

---

### Anomaly Correlation Coefficient (ACC)

```
ACC = correlation(y_true - climatology, y_pred - climatology)
```

**Interpretation**:
- Range: [-1, 1]
  - ACC = 1: Perfect anomaly correlation
  - ACC = 0: No correlation
  - ACC = -1: Perfect negative correlation (worse than useless)
- Measures how well model captures anomalies (deviations from normal)
- **Use when**: Interest in interannual variability rather than absolute values
- **Climate science standard**: ACC > 0.6 considered "skillful"

**Example**: ACC = 0.75 means model anomalies correlate strongly with observed anomalies (good at predicting unusual years)

---

### When to Use Each Metric

| Metric | Primary Use Case | Interpretation Focus |
|--------|------------------|---------------------|
| RMSE   | Model comparison, penalize large errors | Average magnitude with outlier penalty |
| MAE    | Interpretable average error | Simple average error magnitude |
| MAPE   | Scale-independent comparison | Relative error as percentage |
| Skill Score | Demonstrate value over baselines | Improvement percentage |
| ACC    | Anomaly/trend prediction | Correlation of departures from normal |

**Recommendation**: Report **all metrics** in evaluation notebooks. Each provides complementary information.

---

## Baseline Models

### Why Baselines Matter

**Core principle**: A model without baseline comparison is uninterpretable.

- **Context**: RMSE = 0.35 Mkm² means nothing without knowing if persistence achieves 0.50 Mkm² or 0.20 Mkm²
- **Minimum skill threshold**: Models must beat baselines to be useful
- **Scientific integrity**: Prevents claiming success from models that underperform trivial alternatives

---

### Persistence Baseline

**Definition**: Naïve forecast that tomorrow equals today

```
y_pred(t+1) = y_observed(t)
```

**For multi-day horizons**:
```
y_pred(t+h) = y_observed(t)  # Same value for all horizons
```

**Rationale**:
- Simplest possible forecast
- Exploits temporal autocorrelation
- Surprisingly effective for slowly-changing variables (like sea ice extent)
- **Expectation**: Should be relatively easy to beat with proper modeling

**Implementation Requirements**:
- No model fitting needed
- Simply shift time series by forecast horizon
- Evaluate on same test set as all other models

---

### Climatology Baseline

**Definition**: Forecast equals long-term average for that calendar day

```
y_pred(day_of_year) = mean(all_years[day_of_year])
```

**For multi-day horizons**:
```
y_pred(t+h) = climatology[day_of_year(t+h)]
```

**Rationale**:
- Captures seasonal cycle (which dominates Arctic ice extent)
- Ignores interannual variability and trends
- **Expectation**: Should beat persistence during rapid seasonal transitions (spring melt, fall freeze-up)

**Climatology Period Standards**:
- **WMO standard**: 1981-2010 (30-year normal period)
- **Alternative**: Most recent 30 years for non-stationary variables
- **This project**: Document which period used (currently all years 1979-2023 in code; needs consistency with plan)

**Implementation Requirements**:
- Compute day-of-year means from training data only (no test data leakage)
- Handle leap years (366 days)
- Store climatology (can reuse from `ice_extent_climatology` table)

---

### Evaluation Protocol for Baselines

1. **Compute baseline forecasts** on identical test set as other models
2. **Compute metrics** for baselines (RMSE, MAE, MAPE)
   - Note: Skill scores not applicable to baselines (they *are* the baselines)
3. **Use baseline RMSE** to compute skill scores for all other models
4. **Seasonal breakdown**: Baselines may perform differently in winter vs summer
   - Climatology may excel during stable winter maximum
   - Persistence may excel during summer minimum plateau

---

## Denormalization Protocol

### Why Denormalization is Critical

**Problem**: LSTM models trained on normalized data (mean=0, std=1) produce normalized predictions

**Consequence**: Normalized validation loss (e.g., 0.000306) cannot be compared to:
- SARIMA RMSE in Mkm² (e.g., 0.36 Mkm²)
- Baseline metrics in Mkm²
- Physical interpretation (how many million km² off?)

**Solution**: **All** LSTM predictions must be denormalized before evaluation, comparison, or visualization

---

### Denormalization Formula

```python
denormalized_value = (normalized_value × training_std) + training_mean
```

**Requirements**:
1. Use **training set** mean and std (not test set - data leakage!)
2. Apply **same normalization parameters** used during training
3. Denormalize **per-feature** (each feature has own mean/std)

**Example**:
```python
# During training
training_mean = train_data['extent_mkm2'].mean()  # e.g., 10.5 Mkm²
training_std = train_data['extent_mkm2'].std()     # e.g., 3.2 Mkm²
normalized_train = (train_data['extent_mkm2'] - training_mean) / training_std

# During inference
normalized_pred = model.predict(test_input)  # e.g., -0.5 (normalized units)
denormalized_pred = (normalized_pred × training_std) + training_mean
# denormalized_pred = (-0.5 × 3.2) + 10.5 = 8.9 Mkm²
```

---

### Verification Procedure

After denormalization, verify correctness:

1. **Range check**: Denormalized predictions should be in physically plausible range
   - Ice extent: [0, 20] Mkm² approximately
   - Negative values indicate denormalization error

2. **Scale check**: Denormalized RMSE should be comparable to SARIMA RMSE
   - If LSTM denormalized RMSE = 0.000X (way too small), denormalization failed
   - Expected range: 0.1-1.0 Mkm² for reasonable models

3. **Visualization check**: Plot denormalized predictions vs actual values
   - Should overlay actual time series in Mkm²
   - If predictions flat line near zero or mean, suspect normalization issue

---

### Extracting Normalization Parameters

**From Custom Dataset Objects**:
```python
# For CustomArcticDataset or MultivariateArcticDataset
train_dataset = MultivariateArcticDataset(train_data, ...)
mean = train_dataset.mean  # numpy array of shape (n_features,)
std = train_dataset.std    # numpy array of shape (n_features,)

# Target variable (extent_mkm2) is at index train_dataset.target_idx
extent_mean = mean[train_dataset.target_idx]
extent_std = std[train_dataset.target_idx]
```

**From Saved Models**:
- If normalization parameters not saved with model, must recompute from training data
- **Best practice**: Save normalization parameters as model metadata
  - Recommended: JSON file with model checkpoint
  - Example: `best_model_metadata.json` with fields `{"mean": [...], "std": [...]}`

---

## Backtesting Protocol

### Why Backtesting?

**Problem**: Single train/test split can be misleading
- May happen to select easy/hard test period
- No estimate of forecast reliability
- Overfitting risk if hyperparameters tuned on single test set

**Solution**: Multiple train/test splits via time series cross-validation

---

### Expanding Window Cross-Validation

**Approach**: Train on progressively growing history, test on future period

**For this project**:
- **Test years**: 2019, 2020, 2021, 2022, 2023 (5 folds)
- **Training windows**:
  - Fold 1: Train 1989-2018, test 2019
  - Fold 2: Train 1989-2019, test 2020
  - Fold 3: Train 1989-2020, test 2021
  - Fold 4: Train 1989-2021, test 2022
  - Fold 5: Train 1989-2022, test 2023

**Rationale**:
- Expanding window mirrors operational forecasting (always use all available history)
- More realistic than rolling window (which discards old data)
- 5 test years provide 5 independent performance estimates

---

### Rolling Window Alternative

**Approach**: Fixed-size training window slides through time

**Example** (3-year test window):
- Fold 1: Train 1989-2017, test 2018-2020
- Fold 2: Train 1992-2020, test 2021-2023
- etc.

**Use when**:
- Non-stationary data (old data may harm performance)
- Computational constraints (smaller training sets)

**This project**: Expanding window preferred (no evidence Arctic ice physics changed fundamentally)

---

### Seasonal Boundary Handling

**Critical**: Don't split folds mid-winter (Dec/Jan boundary)

**Approach**:
- Use calendar years as fold boundaries (Jan 1 - Dec 31)
- This keeps each winter (Nov-Mar) intact within a single fold

**Rationale**:
- Arctic winter maximum spans Dec-Mar (crosses calendar year)
- Splitting mid-winter creates artificial test dependencies
- Calendar year splits cleanest approach

---

### Metrics Aggregation

**Per-fold metrics**: Compute RMSE, MAE, MAPE, skill scores for each test year

**Aggregate metrics**:
```
Mean RMSE = mean(RMSE across all folds)
Std RMSE = std(RMSE across all folds)
```

**Reporting**:
- Report mean ± std for all metrics
- Example: "RMSE = 0.35 ± 0.08 Mkm²" indicates typical performance and variability

**Statistical significance**:
- Compare model A vs model B using paired t-test on fold-level metrics
- Paired: Each fold provides matched performance estimate
- H0: No difference in mean RMSE between models
- p < 0.05: Reject H0, models significantly different

---

### Backtesting Limitations

**Computational cost**: 5 folds = 5× training time
- LSTM training computationally expensive
- **Acceptable for final evaluation**: Don't backtest during hyperparameter search
- **Alternative**: Single train/test split for exploration, backtest for final model

**Temporal dependencies**: Even expanding window has lookahead information
- Model selection decisions influenced by all test years
- True held-out test: Save final year (2023) completely untouched until very end

---

## Seasonal Analysis

### Why Seasonal Breakdown?

**Rationale**: Arctic sea ice has distinct seasonal regimes with different dynamics

**Seasonal regimes**:
- **Winter (Nov-Mar)**: Maximum extent, slow changes, ice growth
- **Summer (May-Sep)**: Minimum extent, rapid melt, high variability
- **Transitions (Apr, Oct)**: Rapid changes, predictability challenges

**Hypothesis**: Models may perform differently across seasons
- Example: Climatology may excel in stable winter, fail in variable summer
- Example: Atmospheric forcing (ERA5 variables) may matter more during summer melt

---

### Seasonal Definitions

**This Project**:
- **Winter**: November, December, January, February, March (5 months)
- **Summer**: May, June, July, August, September (5 months)
- **Transitions**: April (spring), October (fall) - often excluded from seasonal analysis

**Rationale**:
- Winter: Maximum extent typically March, ice growth Nov-Mar
- Summer: Minimum extent typically September, melt season May-Sep
- Transitions: Brief, high uncertainty periods

**Alternative definitions**: Some studies use meteorological seasons (DJF, MAM, JJA, SON). Document choice.

---

### Evaluation Approach

1. **Split test set** by month into winter and summer subsets
2. **Compute metrics separately** for each season
   - Winter RMSE, MAE, MAPE, skill scores
   - Summer RMSE, MAE, MAPE, skill scores
3. **Compare seasonal performance**:
   - Which models excel in winter vs summer?
   - Are atmospheric variables (ERA5) more important in summer?
   - Does persistence baseline work better in stable winter?

**Visualization**: Heatmap of models × seasons × metrics

---

## Multi-Horizon Evaluation

### Horizon Definitions

**This Project**:
- **1-day ahead**: Standard LSTM single-step forecast
- **7-day ahead**: Seq2Seq LSTM output horizon
- **14-day ahead**: (Planned) Extended Seq2Seq
- **30-day ahead**: (Planned) Monthly forecast

**Horizons of Interest**:
- Operational forecasting: 1-14 days (weather-dependent)
- Seasonal forecasting: 30-90 days (climate-dependent)

---

### Multi-Horizon Model Approaches

**Direct Multi-Step (Seq2Seq)**:
- Single model outputs entire sequence (e.g., 7 days)
- **Advantage**: Learns joint distribution of future sequence
- **Disadvantage**: More complex architecture, harder to train

**Iterated Single-Step**:
- Single-step model applied recursively
  - Predict t+1 from t
  - Predict t+2 from t+1 prediction
  - ... for h steps
- **Advantage**: Simpler model, easier to train
- **Disadvantage**: Error accumulation

---

### Error Accumulation Analysis

**For Seq2Seq models**: Compute RMSE separately for each forecast day

```python
for day in range(1, 8):  # 7-day forecast
    rmse_day = rmse(y_true[:, day-1], y_pred[:, day-1])
    print(f"Day {day} RMSE: {rmse_day:.3f} Mkm²")
```

**Expected pattern**: RMSE increases with horizon
- Day 1 RMSE < Day 2 RMSE < ... < Day 7 RMSE
- Error accumulates due to compounding uncertainty

**Evaluation questions**:
- How fast does error grow? (Linear, exponential, plateau?)
- At what horizon does model stop beating persistence?
- Does Seq2Seq outperform iterated single-step?

**Visualization**: Line plot of RMSE vs forecast day (1-7)

---

### Multi-Horizon Baselines

**Persistence**:
```
y_pred(t+h) = y_observed(t)  # Same for all horizons
```
- Error constant across horizons (no new information used)

**Climatology**:
```
y_pred(t+h) = climatology[day_of_year(t+h)]
```
- Error varies by horizon (depends on climatology accuracy at t+h)

**Comparison**: At what horizon does your model start losing to climatology?
- Short horizon: Model should beat climatology (uses recent observations)
- Long horizon: Model may converge to climatology (loses information)

---

## Statistical Significance Testing

### Why Statistical Tests?

**Problem**: Observed performance differences may be due to chance

**Example**:
- Model A: RMSE = 0.35 Mkm²
- Model B: RMSE = 0.37 Mkm²
- **Question**: Is 0.02 Mkm² difference meaningful or random variation?

**Solution**: Statistical hypothesis testing

---

### Diebold-Mariano Test

**Purpose**: Test whether two forecast models have significantly different accuracy

**Null hypothesis**: Two models have equal forecast accuracy
**Alternative**: Models have different forecast accuracy

**Test statistic**:
```
DM = mean(loss_diff) / se(loss_diff)
```
where:
- `loss_diff = (y - forecast_A)² - (y - forecast_B)²` (squared error differences)
- `se(loss_diff)` = standard error of loss differences (accounting for autocorrelation)

**Implementation**: Available in `scipy.stats` or custom implementation with HAC standard errors

**Interpretation**:
- p < 0.05: Reject H0, models significantly different
- p ≥ 0.05: Fail to reject, no evidence of difference

**Example**:
- Model A vs Model B: p = 0.03 → Model A significantly better
- Model C vs Model D: p = 0.42 → No significant difference

---

### Paired t-Test (Backtesting Context)

**Purpose**: Test whether model A beats model B across multiple test folds

**Approach**:
1. Compute RMSE for both models on each fold (e.g., 5 folds)
2. Compute RMSE differences: `diff = RMSE_A - RMSE_B` for each fold
3. Paired t-test on differences: `t-test(diff, alternative='less')`
   - H0: mean(diff) = 0 (models equal)
   - HA: mean(diff) < 0 (Model A better)

**Advantages**:
- Controls for fold difficulty (paired design)
- Straightforward interpretation

**Limitations**:
- Requires multiple folds (minimum 5, preferably 10+)
- Assumes approximate normality of differences

---

### Effect Size Reporting

**Statistical significance ≠ practical significance**

**Example**:
- Model A: RMSE = 0.350 Mkm²
- Model B: RMSE = 0.349 Mkm²
- p = 0.04 (statistically significant with large sample)
- **But**: 0.001 Mkm² difference likely not operationally meaningful

**Recommendation**: Report both p-values and effect sizes
- **Effect size**: Percentage improvement, skill score difference
- **Example**: "Model A beats Model B by 5% (p=0.03)"

---

## Reporting Standards

### Required Metrics Table

All evaluation notebooks must include standardized comparison table:

| Model | RMSE (Mkm²) | MAE (Mkm²) | MAPE (%) | Skill Score (vs Persistence) | Skill Score (vs Climatology) |
|-------|------------|-----------|----------|------------------------------|------------------------------|
| Persistence | 0.50 | 0.42 | 5.2 | 0.00 | — |
| Climatology | 0.48 | 0.40 | 4.8 | 0.04 | 0.00 |
| SARIMA (raw) | 0.36 | 0.27 | 3.5 | 0.28 | 0.25 |
| SARIMA (anomaly) | 0.41 | 0.33 | 4.0 | 0.18 | 0.15 |
| LSTM (basic) | 0.34* | 0.28* | 3.4* | 0.32* | 0.29* |
| LSTM (multivariate) | 0.31* | 0.25* | 3.1* | 0.38* | 0.35* |
| Seq2Seq (7-day avg) | 0.42* | 0.35* | 4.2* | 0.16* | 0.12* |

*Denormalized from normalized predictions

**Notes**:
- All metrics on identical test set (2020-2023 daily)
- Skill scores relative to baseline RMSEs
- Statistical significance testing results documented separately

---

### Confidence Intervals

**From backtesting**: Report mean ± std
```
RMSE = 0.35 ± 0.08 Mkm² (mean ± std across 5 folds)
```

**From bootstrap**: Resample test set predictions
```
95% CI: [0.31, 0.39] Mkm² (bootstrap percentile method, 1000 iterations)
```

---

### Visualization Standards

**Required plots**:

1. **Time series comparison**: Actual vs all model predictions
   - X-axis: Date (test period)
   - Y-axis: Ice extent (Mkm²)
   - Lines: Actual (thick black), models (colored, varying styles)

2. **Model comparison bar chart**: RMSE by model
   - X-axis: Models
   - Y-axis: RMSE (Mkm²)
   - Error bars: ± std from backtesting
   - Horizontal line: Baseline performance

3. **Seasonal performance heatmap**: Models × Seasons × Metrics
   - Color scale: Performance (green=good, red=bad)
   - Enables quick identification of seasonal strengths

4. **Error distribution boxplots**: Forecast errors by model
   - Shows error spread, outliers, median
   - Complements mean metrics (RMSE/MAE)

5. **(Seq2Seq) Horizon error plot**: RMSE vs forecast day
   - X-axis: Forecast day (1-7)
   - Y-axis: RMSE (Mkm²)
   - Lines: Seq2Seq model, baselines

---

## Quality Checklist

Before considering model evaluation "complete", verify:

- [ ] All models evaluated on **identical test set** (same dates, same samples)
- [ ] LSTM predictions **denormalized** to Mkm² (verified via range/scale checks)
- [ ] **Baseline models** (persistence + climatology) implemented and evaluated
- [ ] **Skill scores** computed vs both baselines
- [ ] **Statistical significance** tested (Diebold-Mariano or paired t-test)
- [ ] **Seasonal breakdown** (winter vs summer) completed
- [ ] **(If applicable) Multi-horizon analysis** for Seq2Seq models
- [ ] **Standardized metrics table** included with all required columns
- [ ] **Visualizations** created (time series, bar chart, heatmap minimum)
- [ ] **Results exported** to `results/model_comparison.csv` with metadata
- [ ] **Interpretation** and recommendations documented

---

## Implementation Roadmap

**Phase 1**: Baseline Implementation (`notebooks/03b_baseline_models.ipynb`)
- Implement persistence and climatology models
- Evaluate on test set (2020-2023)
- Document baseline performance

**Phase 2**: Evaluation Utilities (`src/evaluation_utils.py`)
- Denormalization functions
- Metric computation functions (RMSE, MAE, MAPE, skill scores, ACC)
- Baseline model classes
- Backtesting framework
- Results logging functions

**Phase 3**: Comprehensive Comparison (`notebooks/07_model_comparison.ipynb`)
- Load all models (SARIMA, LSTM, Seq2Seq)
- Denormalize LSTM predictions
- Compute standardized metrics
- Statistical significance testing
- Seasonal analysis
- Visualization generation
- Results export

**Phase 4**: Backtesting (optional enhancement)
- Implement expanding window CV
- Retrain models on multiple folds
- Aggregate metrics with confidence intervals

---

## References

**Forecast Evaluation**:
- Wilks, D. S. (2011). Statistical Methods in the Atmospheric Sciences (3rd ed.). Academic Press.
- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.

**Diebold-Mariano Test**:
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.

**Skill Scores**:
- Murphy, A. H. (1988). Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. Monthly Weather Review, 116(12), 2417-2424.

**Related Project Documentation**:
- `docs/methodology.md`: Implementation details for all models
- `docs/data_dictionary.md`: Database schema and data specifications
- `docs/project_plan.md`: Project phases and milestones

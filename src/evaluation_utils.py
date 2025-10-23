"""
Evaluation Utilities for Arctic Sea Ice Extent Forecasting

This module provides standardized evaluation tools for comparing forecast models:
- Denormalization functions for LSTM predictions
- Standard metrics (RMSE, MAE, MAPE, skill scores, ACC)
- Baseline model classes (Persistence, Climatology)
- Backtesting framework (expanding window cross-validation)
- Results logging and comparison utilities

All evaluation follows protocols documented in docs/evaluation_methodology.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json


# =============================================================================
# DENORMALIZATION FUNCTIONS
# =============================================================================

def denormalize(normalized_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Convert normalized predictions back to original units.

    Args:
        normalized_values: Array of normalized predictions (mean=0, std=1)
        mean: Training set mean used for normalization
        std: Training set standard deviation used for normalization

    Returns:
        Array of denormalized values in original units (Mkm² for ice extent)

    Example:
        >>> normalized_pred = np.array([-0.5, 0.0, 0.5])
        >>> mean, std = 10.5, 3.2
        >>> denormalize(normalized_pred, mean, std)
        array([ 8.9, 10.5, 12.1])
    """
    return (normalized_values * std) + mean


def get_normalization_params(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract normalization parameters from CustomArcticDataset or MultivariateArcticDataset.

    Args:
        dataset: Dataset object with .mean and .std attributes

    Returns:
        Tuple of (mean, std) as numpy arrays

    Example:
        >>> from notebooks import CustomArcticDataset
        >>> train_dataset = CustomArcticDataset(train_data, ...)
        >>> mean, std = get_normalization_params(train_dataset)
    """
    return dataset.mean, dataset.std


def denormalize_target(normalized_predictions: np.ndarray,
                       dataset,
                       target_name: str = 'extent_mkm2') -> np.ndarray:
    """
    Denormalize predictions for target variable from dataset.

    Args:
        normalized_predictions: Normalized predictions from LSTM model
        dataset: Dataset object with normalization parameters
        target_name: Name of target variable (default: 'extent_mkm2')

    Returns:
        Denormalized predictions in original units (Mkm²)

    Example:
        >>> predictions_normalized = model.predict(test_loader)
        >>> predictions_mkm2 = denormalize_target(predictions_normalized, train_dataset)
    """
    mean, std = get_normalization_params(dataset)

    # Find target variable index
    target_idx = dataset.target_idx if hasattr(dataset, 'target_idx') else 0

    # Extract target mean and std
    target_mean = mean[target_idx] if mean.ndim > 0 else mean
    target_std = std[target_idx] if std.ndim > 0 else std

    return denormalize(normalized_predictions, target_mean, target_std)


# =============================================================================
# STANDARD METRICS
# =============================================================================

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value (same units as inputs)

    Formula:
        RMSE = sqrt(mean((y_true - y_pred)²))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value (same units as inputs)

    Formula:
        MAE = mean(|y_true - y_pred|)
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Args:
        y_true: True values (must be non-zero)
        y_pred: Predicted values

    Returns:
        MAPE as percentage (0-100+)

    Formula:
        MAPE = mean(|y_true - y_pred| / |y_true|) × 100

    Note:
        Returns np.nan if any y_true values are zero (MAPE undefined)
    """
    if np.any(y_true == 0):
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_skill_score(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_baseline: np.ndarray) -> float:
    """
    Compute skill score relative to baseline model.

    Args:
        y_true: True values
        y_pred: Model predictions
        y_baseline: Baseline predictions (e.g., persistence or climatology)

    Returns:
        Skill score (unitless)
        - SS = 1: Perfect forecast
        - SS = 0: No better than baseline
        - SS > 0: Model beats baseline
        - SS < 0: Model worse than baseline

    Formula:
        SS = 1 - (RMSE_model / RMSE_baseline)

    Example:
        >>> y_true = np.array([10.5, 11.2, 9.8])
        >>> y_pred = np.array([10.3, 11.0, 9.9])  # RMSE = 0.17
        >>> y_baseline = np.array([10.0, 10.0, 10.0])  # RMSE = 0.62
        >>> compute_skill_score(y_true, y_pred, y_baseline)
        0.73  # 73% improvement over baseline
    """
    rmse_model = compute_rmse(y_true, y_pred)
    rmse_baseline = compute_rmse(y_true, y_baseline)

    if rmse_baseline == 0:
        return np.nan  # Avoid division by zero

    return 1 - (rmse_model / rmse_baseline)


def compute_anomaly_correlation(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                climatology: np.ndarray) -> float:
    """
    Compute Anomaly Correlation Coefficient (ACC).

    Args:
        y_true: True values
        y_pred: Predicted values
        climatology: Climatological baseline for same dates

    Returns:
        ACC value (-1 to 1)
        - ACC = 1: Perfect anomaly correlation
        - ACC = 0: No correlation
        - ACC < 0: Negative correlation (bad)

    Formula:
        ACC = correlation(y_true - climatology, y_pred - climatology)

    Note:
        ACC measures how well model captures departures from normal
    """
    anomaly_true = y_true - climatology
    anomaly_pred = y_pred - climatology

    return np.corrcoef(anomaly_true, anomaly_pred)[0, 1]


def compute_all_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_baseline_persistence: Optional[np.ndarray] = None,
                       y_baseline_climatology: Optional[np.ndarray] = None,
                       climatology: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all standard evaluation metrics.

    Args:
        y_true: True values
        y_pred: Model predictions
        y_baseline_persistence: Persistence baseline predictions (optional)
        y_baseline_climatology: Climatology baseline predictions (optional)
        climatology: Climatological values for ACC computation (optional)

    Returns:
        Dictionary with all computed metrics

    Example:
        >>> metrics = compute_all_metrics(
        ...     y_true=test_actual,
        ...     y_pred=model_predictions,
        ...     y_baseline_persistence=persistence_forecast,
        ...     y_baseline_climatology=climatology_forecast,
        ...     climatology=climatology_values
        ... )
        >>> print(f"RMSE: {metrics['rmse']:.3f} Mkm²")
        >>> print(f"Skill vs Persistence: {metrics['skill_score_persistence']:.2%}")
    """
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred)
    }

    if y_baseline_persistence is not None:
        metrics['skill_score_persistence'] = compute_skill_score(
            y_true, y_pred, y_baseline_persistence
        )

    if y_baseline_climatology is not None:
        metrics['skill_score_climatology'] = compute_skill_score(
            y_true, y_pred, y_baseline_climatology
        )

    if climatology is not None:
        metrics['anomaly_correlation'] = compute_anomaly_correlation(
            y_true, y_pred, climatology
        )

    return metrics


# =============================================================================
# BASELINE MODEL CLASSES
# =============================================================================

class PersistenceModel:
    """
    Persistence baseline: Forecast equals most recent observation.

    Naïve forecast: y_pred(t+h) = y_observed(t)

    This is the simplest possible forecast and exploits temporal autocorrelation.
    All models should beat this baseline to be considered useful.

    Example:
        >>> model = PersistenceModel()
        >>> model.fit(X_train, y_train)  # No actual training needed
        >>> predictions = model.predict(X_test, horizon=1)
    """

    def __init__(self):
        """Initialize persistence model."""
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit persistence model (no-op, included for sklearn compatibility).

        Args:
            X: Features (not used)
            y: Target values (not used)

        Returns:
            self
        """
        self.is_fitted = True
        return self

    def predict(self, X, horizon: int = 1):
        """
        Generate persistence forecasts.

        Args:
            X: Last observed values (shape: (n_samples,))
            horizon: Forecast horizon in days (not used, persistence same for all horizons)

        Returns:
            Predictions array (same as X)

        Note:
            For persistence, forecast at all horizons equals last observation
        """
        return np.array(X)


class ClimatologyModel:
    """
    Climatology baseline: Forecast equals long-term day-of-year average.

    Forecast: y_pred(day_of_year) = mean(all_years[day_of_year])

    Captures seasonal cycle but ignores interannual variability and trends.
    Useful baseline during rapid seasonal transitions.

    Example:
        >>> model = ClimatologyModel()
        >>> model.fit(train_dates, train_values)
        >>> predictions = model.predict(test_dates)
    """

    def __init__(self):
        """Initialize climatology model."""
        self.climatology = None
        self.is_fitted = False

    def fit(self, dates: pd.Series, values: pd.Series):
        """
        Compute day-of-year climatology from training data.

        Args:
            dates: Pandas Series of dates
            values: Pandas Series of ice extent values (Mkm²)

        Returns:
            self

        Note:
            Computes 366 climatology values (one per day including leap day)
        """
        df = pd.DataFrame({'date': dates, 'value': values})
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear

        # Compute mean for each day of year
        self.climatology = df.groupby('day_of_year')['value'].mean()

        self.is_fitted = True
        return self

    def predict(self, dates: pd.Series) -> np.ndarray:
        """
        Generate climatology forecasts for given dates.

        Args:
            dates: Pandas Series of forecast dates

        Returns:
            Array of climatology predictions (Mkm²)

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")

        day_of_year = pd.to_datetime(dates).dt.dayofyear
        return self.climatology[day_of_year].values


# =============================================================================
# BACKTESTING FRAMEWORK
# =============================================================================

class TimeSeriesExpandingWindow:
    """
    Expanding window cross-validation for time series.

    Progressively grows training window, tests on future period.
    Mirrors operational forecasting (always use all available history).

    Example:
        >>> cv = TimeSeriesExpandingWindow(test_years=[2019, 2020, 2021, 2022, 2023])
        >>> for train_idx, test_idx in cv.split(data, date_column='date'):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
        ...     # Train and evaluate model
    """

    def __init__(self, test_years: List[int]):
        """
        Initialize expanding window cross-validator.

        Args:
            test_years: List of years to use as test sets
                Example: [2019, 2020, 2021, 2022, 2023]
        """
        self.test_years = sorted(test_years)

    def split(self, data: pd.DataFrame, date_column: str = 'date'):
        """
        Generate train/test splits for each test year.

        Args:
            data: DataFrame with time series data
            date_column: Name of column containing dates

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Example:
            Fold 1: Train 1989-2018, Test 2019
            Fold 2: Train 1989-2019, Test 2020
            Fold 3: Train 1989-2020, Test 2021
            etc.
        """
        dates = pd.to_datetime(data[date_column])

        for test_year in self.test_years:
            # Train on all data before test year
            train_mask = dates.dt.year < test_year
            test_mask = dates.dt.year == test_year

            train_idx = data.index[train_mask].tolist()
            test_idx = data.index[test_mask].tolist()

            if len(test_idx) > 0:  # Only yield if test set non-empty
                yield train_idx, test_idx


def run_backtesting(model,
                   data: pd.DataFrame,
                   cv_splitter: TimeSeriesExpandingWindow,
                   feature_cols: List[str],
                   target_col: str,
                   date_col: str = 'date') -> pd.DataFrame:
    """
    Run model through backtesting splits and collect metrics.

    Args:
        model: Model with .fit() and .predict() methods
        data: DataFrame with features, target, and dates
        cv_splitter: TimeSeriesExpandingWindow object
        feature_cols: List of feature column names
        target_col: Target column name
        date_col: Date column name

    Returns:
        DataFrame with metrics for each fold
        Columns: fold, test_year, rmse, mae, mape, n_samples

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> cv = TimeSeriesExpandingWindow([2019, 2020, 2021, 2022, 2023])
        >>> results = run_backtesting(
        ...     model=Ridge(),
        ...     data=df,
        ...     cv_splitter=cv,
        ...     feature_cols=['extent_lag1', 't2m_mean'],
        ...     target_col='extent_mkm2'
        ... )
        >>> print(f"Mean RMSE: {results['rmse'].mean():.3f} ± {results['rmse'].std():.3f}")
    """
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(data, date_col)):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Extract features and target
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = {
            'fold': fold_idx,
            'test_year': test_data[date_col].dt.year.iloc[0],
            'rmse': compute_rmse(y_test, y_pred),
            'mae': compute_mae(y_test, y_pred),
            'mape': compute_mape(y_test, y_pred),
            'n_samples': len(y_test)
        }

        results.append(metrics)

    return pd.DataFrame(results)


# =============================================================================
# RESULTS TRACKING
# =============================================================================

def log_model_results(model_name: str,
                     metrics: Dict[str, float],
                     scale: str = 'daily',
                     metadata: Optional[Dict] = None,
                     output_file: Union[str, Path] = 'results/model_comparison.csv') -> None:
    """
    Append model performance to CSV log file.

    Args:
        model_name: Descriptive name for model (e.g., "LSTM_multivariate_lagged")
        metrics: Dictionary of metric_name: value
        scale: Temporal scale of predictions ('daily', 'monthly', 'weekly', 'multi-day', etc.)
        metadata: Optional dictionary of model metadata (hyperparameters, etc.)
        output_file: Path to CSV file (created if doesn't exist)

    Example:
        >>> metrics = {
        ...     'rmse': 0.35,
        ...     'mae': 0.28,
        ...     'mape': 3.4,
        ...     'skill_score_persistence': 0.32
        ... }
        >>> metadata = {'hidden_size': 64, 'lookback': 30, 'features': 'multivariate_lagged'}
        >>> log_model_results('LSTM_lagged', metrics, scale='daily', metadata=metadata)

    Note:
        The 'scale' parameter helps filter and compare models at the same temporal resolution:
        - 'daily': Single-day forecasts (horizon=1 day)
        - 'monthly': Monthly aggregated forecasts
        - 'weekly' or 'multi-day': Multi-step forecasts (e.g., 7-day horizon)
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare row
    row = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'scale': scale,
        **metrics
    }

    # Add metadata as JSON string
    if metadata:
        row['metadata'] = json.dumps(metadata)

    # Create DataFrame
    df_new = pd.DataFrame([row])

    # Append to existing file or create new
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_file, index=False)
    print(f"✓ Logged results for {model_name} to {output_file}")


def load_results(results_file: Union[str, Path] = 'results/model_comparison.csv') -> pd.DataFrame:
    """
    Load model comparison results from CSV.

    Args:
        results_file: Path to CSV file

    Returns:
        DataFrame with all logged results

    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    results_file = Path(results_file)

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return pd.read_csv(results_file)


def create_comparison_table(results_df: pd.DataFrame,
                           sort_by: str = 'rmse',
                           ascending: bool = True,
                           filter_scale: Optional[str] = None) -> pd.DataFrame:
    """
    Create formatted comparison table from results.

    Args:
        results_df: DataFrame from load_results()
        sort_by: Column to sort by (default: 'rmse')
        ascending: Sort order (default: True for metrics where lower is better)
        filter_scale: Optional filter by temporal scale ('daily', 'monthly', etc.)

    Returns:
        Formatted DataFrame with models ranked by performance

    Example:
        >>> results = load_results()
        >>> # Compare all daily models
        >>> table = create_comparison_table(results, sort_by='rmse', filter_scale='daily')
        >>> print(table.to_markdown(index=False))
        >>>
        >>> # Compare all monthly models
        >>> table_monthly = create_comparison_table(results, filter_scale='monthly')
        >>> print(table_monthly)
    """
    # Apply scale filter if specified
    if filter_scale and 'scale' in results_df.columns:
        results_df = results_df[results_df['scale'] == filter_scale].copy()

    # Select columns for comparison
    metric_cols = ['rmse', 'mae', 'mape', 'skill_score_persistence',
                   'skill_score_climatology', 'anomaly_correlation']

    base_cols = ['model_name']
    if 'scale' in results_df.columns:
        base_cols.append('scale')

    available_cols = base_cols + [col for col in metric_cols if col in results_df.columns]

    table = results_df[available_cols].copy()

    # Get most recent result for each model (if multiple runs)
    if 'timestamp' in results_df.columns:
        table['timestamp'] = pd.to_datetime(results_df['timestamp'])
        groupby_cols = ['model_name', 'scale'] if 'scale' in table.columns else ['model_name']
        table = table.sort_values('timestamp').groupby(groupby_cols).last().reset_index()
        table = table.drop('timestamp', axis=1)

    # Sort by specified metric
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=ascending)

    return table


# =============================================================================
# SEASONAL ANALYSIS UTILITIES
# =============================================================================

def split_by_season(data: pd.DataFrame,
                   date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into winter and summer seasons.

    Args:
        data: DataFrame with date column
        date_col: Name of date column

    Returns:
        Tuple of (winter_data, summer_data)
        - Winter: November, December, January, February, March
        - Summer: May, June, July, August, September

    Note:
        Spring (April) and fall (October) excluded from seasonal analysis
    """
    data = data.copy()
    data['month'] = pd.to_datetime(data[date_col]).dt.month

    winter_months = [11, 12, 1, 2, 3]
    summer_months = [5, 6, 7, 8, 9]

    winter_data = data[data['month'].isin(winter_months)].drop('month', axis=1)
    summer_data = data[data['month'].isin(summer_months)].drop('month', axis=1)

    return winter_data, summer_data


def compute_seasonal_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            dates: pd.Series,
                            y_baseline_persistence: Optional[np.ndarray] = None,
                            y_baseline_climatology: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for winter and summer seasons.

    Args:
        y_true: True values
        y_pred: Model predictions
        dates: Pandas Series of dates
        y_baseline_persistence: Persistence baseline (optional)
        y_baseline_climatology: Climatology baseline (optional)

    Returns:
        Dictionary with 'winter' and 'summer' keys, each containing metrics dict

    Example:
        >>> seasonal_metrics = compute_seasonal_metrics(
        ...     y_true=test_actual,
        ...     y_pred=model_predictions,
        ...     dates=test_dates,
        ...     y_baseline_persistence=persistence_forecast
        ... )
        >>> print(f"Winter RMSE: {seasonal_metrics['winter']['rmse']:.3f}")
        >>> print(f"Summer RMSE: {seasonal_metrics['summer']['rmse']:.3f}")
    """
    df = pd.DataFrame({
        'date': dates,
        'y_true': y_true,
        'y_pred': y_pred
    })

    if y_baseline_persistence is not None:
        df['y_baseline_persistence'] = y_baseline_persistence
    if y_baseline_climatology is not None:
        df['y_baseline_climatology'] = y_baseline_climatology

    winter_data, summer_data = split_by_season(df, date_col='date')

    results = {}

    for season_name, season_data in [('winter', winter_data), ('summer', summer_data)]:
        if len(season_data) == 0:
            results[season_name] = {}
            continue

        season_metrics = compute_all_metrics(
            y_true=season_data['y_true'].values,
            y_pred=season_data['y_pred'].values,
            y_baseline_persistence=season_data['y_baseline_persistence'].values if 'y_baseline_persistence' in season_data else None,
            y_baseline_climatology=season_data['y_baseline_climatology'].values if 'y_baseline_climatology' in season_data else None
        )

        results[season_name] = season_metrics

    return results


if __name__ == '__main__':
    # Example usage and verification
    print("Evaluation Utilities Module")
    print("=" * 60)

    # Test denormalization
    print("\n1. Testing denormalization:")
    normalized = np.array([-0.5, 0.0, 0.5])
    mean, std = 10.5, 3.2
    denorm = denormalize(normalized, mean, std)
    print(f"   Normalized: {normalized}")
    print(f"   Denormalized: {denorm} (mean={mean}, std={std})")

    # Test metrics
    print("\n2. Testing metrics:")
    y_true = np.array([10.5, 11.2, 9.8, 10.1, 11.5])
    y_pred = np.array([10.3, 11.0, 9.9, 10.2, 11.3])
    y_baseline = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    metrics = compute_all_metrics(y_true, y_pred, y_baseline, y_baseline)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    # Test baseline models
    print("\n3. Testing baseline models:")
    dates = pd.date_range('2020-01-01', periods=5)
    values = np.array([10.5, 11.2, 9.8, 10.1, 11.5])

    persistence = PersistenceModel()
    persistence.fit(None, None)
    print(f"   Persistence prediction: {persistence.predict(values[-1:], horizon=1)[0]:.2f}")

    climatology = ClimatologyModel()
    train_dates = pd.date_range('2015-01-01', '2019-12-31')
    train_values = np.random.normal(10.0, 2.0, len(train_dates))
    climatology.fit(train_dates, train_values)
    print(f"   Climatology fitted on {len(train_dates)} samples")

    print("\n✓ All tests passed. Module ready for use.")

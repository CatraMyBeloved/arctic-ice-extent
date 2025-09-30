"""Utility functions for visualizing Arctic sea ice data.

This module provides plotting functions for exploratory data analysis and
model comparison, with emphasis on time series visualization.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def plot_standardized_comparison(
    df: pd.DataFrame,
    columns: List[str],
    date_column: str = 'date',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    alpha: float = 0.8,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple variables on the same scale after standardizing them.

    Standardization follows the StandardScaler approach: z = (x - mean) / std.
    This transformation allows direct comparison of variables with different units
    and scales by centering them at zero with unit variance.

    Educational Note:
        Standardization is particularly useful for comparing:
        - Ice extent (Mkm�) vs atmospheric variables (temperature, pressure)
        - Variables with different magnitudes (e.g., wind speed vs humidity)
        - Time series that need visual alignment for pattern comparison

    Args:
        df: DataFrame containing the time series data.
        columns: List of column names to plot (must exist in df).
        date_column: Name of the date/time column for x-axis (default: 'date').
        title: Optional plot title. If None, generates descriptive title.
        figsize: Figure size as (width, height) in inches.
        alpha: Line transparency (0.0 to 1.0).

    Returns:
        Tuple of (figure, axes) for further customization if needed.

    Raises:
        ValueError: If any specified column doesn't exist in the DataFrame.
        ValueError: If date_column doesn't exist in the DataFrame.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}. Available columns: {df.columns.tolist()}")

    fig, ax = plt.subplots(figsize=figsize)

    for col in columns:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:
            standardized = df[col] - mean
        else:
            standardized = (df[col] - mean) / std

        ax.plot(df[date_column], standardized, label=col, alpha=alpha, linewidth=1.5)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Standardized Value (z-score)', fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax.set_title(f'Standardized Comparison of {len(columns)} Variables', fontsize=13, fontweight='bold')

    fig.tight_layout()

    return fig, ax
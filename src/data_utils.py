import pandas as pd
from pathlib import Path
from typing import Optional

# Define project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load data from the raw data directory.

    Args:
        filename: Name of the file in data/raw/

    Returns:
        DataFrame with the loaded data

    Example:
        df = load_raw_data("sales_data.csv")
    """
    filepath = DATA_DIR / "raw" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Handle different file types
    if filename.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filename.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filename.endswith('.parquet'):
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to the processed directory.

    Args:
        df: DataFrame to save
        filename: Name for the output file

    Example:
        save_processed_data(cleaned_df, "sales_cleaned.parquet")
    """
    output_path = DATA_DIR / "processed" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create dir if needed

    if filename.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    elif filename.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {filename}")

    print(f"Data saved to {output_path}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a standard summary of a DataFrame.

    Returns:
        Dictionary with shape, dtypes, missing values, etc.
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
    }
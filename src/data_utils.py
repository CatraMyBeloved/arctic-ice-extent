"""Utility functions for loading and processing Arctic sea ice data.

This module provides functions for loading data from PostgreSQL database and
merging ERA5 atmospheric data with sea ice extent measurements for analysis.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Define project paths
PROJECT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_DIR / "data"
DATABASE_URL: str = 'postgresql://postgres:password@localhost:5432/seaice'


def load_yearly_data_from_database(year: int, region: Optional[str] = None) -> pd.DataFrame:
    """Load yearly sea ice data from the PostgreSQL database.

    Args:
        year: The year for which to load data.
        region: The region to filter data. If None, load all regions.

    Returns:
        DataFrame containing the sea ice data for the specified year and region.
    """
    import sqlalchemy

    # Create a database connection
    engine = sqlalchemy.create_engine(DATABASE_URL)

    # Handle pan-arctic region by loading from pan-arctic table
    if region == 'pan_arctic':
        query = f"SELECT date, 'pan_arctic' as region, extent_mkm2 as extent FROM ice_extent_pan_arctic_daily WHERE EXTRACT(YEAR FROM date) = {year}"
    else:
        # Construct SQL query for regional data
        query = f"SELECT * FROM ice_extent_regional_daily WHERE EXTRACT(YEAR FROM date) = {year}"
        if region:
            query += f" AND region = '{region}'"

    # Load data into DataFrame
    df = pd.read_sql(query, engine)

    return df

def load_data_for_year(year: int, region: str) -> pd.DataFrame:
    """Load and merge ERA5 atmospheric data with sea ice extent data for a specific year and region.

    Handles the format mismatch between long-format parquet files and wide-format database tables
    by pivoting the atmospheric data and mapping region names between datasets.

    Args:
        year: The year for which to load the data.
        region: The target region name (using database naming convention).

    Returns:
        Merged dataset with atmospheric features as columns and ice extent data.

    Raises:
        FileNotFoundError: If the parquet file for the specified year does not exist.
        ValueError: If no data is available for the specified region.
    """
    region_mapping = {
        'Central_Arctic': 'Central',
        'Chukchi-NA': 'Chukchi',
        'Chukchi-Asia': 'Chukchi',
        'Bering-NA': 'Bering',
        'Bering-Asia': 'Bering',
        'Can_Arch': 'CanadianArchipelago',
        'E_Greenland': 'Greenland',
        'E_Siberian': 'East',
        'Baffin': 'Baffin',
        'Barents': 'Barents',
        'Beaufort': 'Beaufort',
        'Hudson': 'Hudson',
        'Kara': 'Kara',
        'Laptev': 'Laptev',
        'Okhotsk': 'Okhotsk',
        'pan_arctic': 'pan_arctic'  # Pan-arctic maps to itself
    }

    parquet_filepath = f"../data/processed/parquet/era5_regional_{year}.parquet"

    if not os.path.exists(parquet_filepath):
        raise FileNotFoundError(f"No data found for year {year} at {parquet_filepath}")

    df_atmospheric = pd.read_parquet(parquet_filepath)

    # Handle pan-arctic region - it maps directly
    if region == 'pan_arctic':
        df_filtered = df_atmospheric[df_atmospheric['region'] == 'pan_arctic'].copy()
        df_filtered['mapped_region'] = 'pan_arctic'
    else:
        # Handle regional data with mapping
        df_atmospheric['mapped_region'] = df_atmospheric['region'].map(region_mapping)
        df_filtered = df_atmospheric[df_atmospheric['mapped_region'] == region].copy()

    if df_filtered.empty:
        available_regions = df_atmospheric['region'].unique() if region == 'pan_arctic' else df_atmospheric['mapped_region'].dropna().unique()
        raise ValueError(f"No data found for region '{region}'. Available regions: {sorted(available_regions)}")

    df_filtered['var_stat'] = df_filtered['variable'] + '_' + df_filtered['stat']
    df_wide = df_filtered.pivot_table(
        index=['date', 'mapped_region'],
        columns='var_stat',
        values='value',
        aggfunc='first'
    ).reset_index()

    df_wide = df_wide.rename(columns={'mapped_region': 'region'})
    df_wide.columns.name = None

    df_ice_extent = load_yearly_data_from_database(year, region)
    df_merged = pd.merge(df_wide, df_ice_extent, on=['date', 'region'], how='left')

    return df_merged
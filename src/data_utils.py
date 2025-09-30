"""Utility functions for loading and processing Arctic sea ice data.

This module provides functions for loading data from PostgreSQL database and
merging ERA5 atmospheric data with sea ice extent measurements for analysis.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Define project paths
PROJECT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_DIR / "data"
DATABASE_URL: str = 'postgresql://postgres:password@localhost:5432/seaice'


def load_yearly_data_from_database(year: int, region: Optional[str] = None) -> pd.DataFrame:
    """Load yearly sea ice data from the PostgreSQL database.

    Returns data with consistent column naming: date, region, extent_mkm2.
    All extent values are in million km² (Mkm²) across both pan-arctic and regional tables.

    Args:
        year: The year for which to load data.
        region: The region to filter data. If None, load all regions.

    Returns:
        DataFrame with columns: date, region, extent_mkm2 (all in Mkm² units).
    """
    import sqlalchemy

    # Create a database connection
    engine = sqlalchemy.create_engine(DATABASE_URL)

    # Handle pan-arctic region by loading from pan-arctic table
    if region == 'pan_arctic':
        query = f"SELECT date, region, extent_mkm2 FROM ice_extent_pan_arctic_daily WHERE EXTRACT(YEAR FROM date) = {year}"
    else:
        # Construct SQL query for regional data
        query = f"SELECT date, region, extent_mkm2 FROM ice_extent_regional_daily WHERE EXTRACT(YEAR FROM date) = {year}"
        if region:
            query += f" AND region = '{region}'"

    # Load data into DataFrame
    df = pd.read_sql(query, engine)

    return df

def _load_data_for_year(year: int, region: str) -> pd.DataFrame:
    """Internal helper: Load and merge ERA5 atmospheric data with sea ice extent for a year and region.

    Handles the format mismatch between long-format parquet files and wide-format database tables
    by pivoting the atmospheric data and mapping region names between datasets.

    Args:
        year: The year for which to load the data.
        region: The target region name (using database naming convention).

    Returns:
        Merged dataset with atmospheric features as columns and extent_mkm2 in Mkm².

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

    parquet_filepath = DATA_DIR / "processed" / "parquet" / f"era5_regional_{year}.parquet"

    if not parquet_filepath.exists():
        raise FileNotFoundError(f"No data found for year {year} at {parquet_filepath}")

    df_atmospheric = pd.read_parquet(parquet_filepath)

    if region == 'pan_arctic':
        df_filtered = df_atmospheric[df_atmospheric['region'] == 'pan_arctic'].copy()
        df_filtered['mapped_region'] = 'pan_arctic'
    else:
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


def load_data(regions: Union[str, List[str]], years: Union[int, List[int]]) -> pd.DataFrame:
    """Load ice extent and atmospheric data for multiple regions and years.

    This is the main public API for loading combined ERA5 atmospheric and sea ice extent data.
    Accepts flexible inputs (single values or lists) and returns a unified DataFrame with
    consistent units (extent_mkm2 in million km²).

    Args:
        regions: Single region name or list of regions (e.g., 'Central', ['Barents', 'Beaufort']).
                 Use 'pan_arctic' for full Arctic coverage.
        years: Single year or list of years (e.g., 2000, [2000, 2001, 2002], range(2000, 2024)).

    Returns:
        DataFrame with columns including: date, region, extent_mkm2, and ERA5 variables
        (t2m_mean, msl_mean, tp_mean, wind_speed_mean, etc.).

    Raises:
        FileNotFoundError: If ERA5 parquet files don't exist for specified years.
        ValueError: If specified regions don't exist in the data.

    """
    if isinstance(regions, str):
        regions = [regions]
    if isinstance(years, int):
        years = [years]

    years = list(years)
    regions = list(regions)

    dataframes = []
    for year in years:
        for region in regions:
            try:
                df_year_region = _load_data_for_year(year, region)
                dataframes.append(df_year_region)
            except (FileNotFoundError, ValueError) as e:
                raise type(e)(f"Error loading data for year {year}, region '{region}': {str(e)}") from e

    if not dataframes:
        raise ValueError(f"No data loaded for regions {regions} and years {years}")

    df_combined = pd.concat(dataframes, ignore_index=True)

    df_combined = df_combined.sort_values(['date', 'region']).reset_index(drop=True)

    return df_combined
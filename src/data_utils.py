import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd

# Define project paths
PROJECT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_DIR / "data"
DATABASE_URL: str = 'postgresql://postgres:password@localhost:5432/seaice'

# Valid region names (database naming convention)
Region = Literal[
    'pan_arctic',
    'Baffin',
    'Barents',
    'Beaufort',
    'Bering',
    'CanadianArchipelago',
    'Central',
    'East',
    'Greenland',
    'Hudson',
    'Kara',
    'Laptev',
    'Okhotsk',
]

# All available regions as a list (for "all" parameter support)
ALL_REGIONS: List[str] = [
    'pan_arctic', 'Baffin', 'Barents', 'Beaufort', 'Bering',
    'CanadianArchipelago', 'Central', 'East', 'Greenland',
    'Hudson', 'Kara', 'Laptev', 'Okhotsk'
]

# Public API exports
__all__ = ['load_data', 'load_yearly_data_from_database', 'Region', 'get_available_regions', 'get_available_years']


def get_available_regions() -> List[str]:
    """Get list of all available regions.

    Returns:
        List of all valid region names that can be used with load_data().
    """
    return ALL_REGIONS.copy()


def get_available_years() -> List[int]:
    """Get list of all available years based on parquet files.

    Returns:
        List of years for which data is available.
    """
    parquet_dir = DATA_DIR / "processed" / "parquet"
    if not parquet_dir.exists():
        return []

    years = []
    for file in parquet_dir.glob("era5_regional_*.parquet"):
        try:
            year = int(file.stem.split('_')[-1])
            years.append(year)
        except ValueError:
            continue

    return sorted(years)


def load_yearly_data_from_database(year: int, region: Optional[Region] = None) -> pd.DataFrame:
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
    df = pd.read_sql(query, engine, parse_dates=['date'])

    return df

def _load_data_for_year(year: int, region: Region) -> pd.DataFrame:
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

    # Ensure datetime types for proper merging
    df_wide['date'] = pd.to_datetime(df_wide['date'])

    df_ice_extent = load_yearly_data_from_database(year, region)

    # Ensure both dataframes have consistent datetime types before merging
    df_ice_extent['date'] = pd.to_datetime(df_ice_extent['date'])

    df_merged = pd.merge(df_wide, df_ice_extent, on=['date', 'region'], how='left')

    return df_merged


def _interpolate_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing dates in the ice extent data.

    Handles the historical issue where ice extent data was recorded every other day
    from 1979-1989. Creates a complete daily time series by linearly interpolating
    missing extent_mkm2 values. Atmospheric data (ERA5) is forward-filled if missing.

    Args:
        df: DataFrame with date, region, extent_mkm2, and atmospheric columns.

    Returns:
        DataFrame with daily resolution and interpolated extent values.
    """
    interpolated_groups = []

    for region, group in df.groupby('region'):
        # Create complete date range for this region
        date_range = pd.date_range(
            start=group['date'].min(),
            end=group['date'].max(),
            freq='D'
        )

        # Reindex to include all dates
        group = group.set_index('date').reindex(date_range)

        # Fill region column (it becomes NaN on new rows)
        group['region'] = region

        # Interpolate ice extent linearly
        group['extent_mkm2'] = group['extent_mkm2'].interpolate(method='linear')

        # Forward-fill atmospheric data (ERA5 should be complete, but handle edge cases)
        atmospheric_cols = [col for col in group.columns if col not in ['region', 'extent_mkm2']]
        for col in atmospheric_cols:
            if group[col].isna().any():
                group[col] = group[col].ffill()

        group = group.reset_index().rename(columns={'index': 'date'})
        interpolated_groups.append(group)

    return pd.concat(interpolated_groups, ignore_index=True).sort_values(['date', 'region']).reset_index(drop=True)


def load_data(
    regions: Union[Region, List[Region], Literal['all']] = 'all',
    years: Union[int, List[int], range, Literal['all']] = 'all'
) -> pd.DataFrame:
    """Load ice extent and atmospheric data for multiple regions and years.

    This is the main public API for loading combined ERA5 atmospheric and sea ice extent data.
    Accepts flexible inputs (single values, lists, or 'all') and returns a unified DataFrame with
    consistent units (extent_mkm2 in million km²).

    Args:
        regions: Region(s) to load. Options:
                 - 'all': Load all available regions (default)
                 - Single region: 'Central', 'Bering', etc.
                 - List of regions: ['Barents', 'Beaufort', 'Central']
                 Valid regions: pan_arctic, Baffin, Barents, Beaufort, Bering, CanadianArchipelago,
                 Central, East, Greenland, Hudson, Kara, Laptev, Okhotsk
        years: Year(s) to load. Options:
               - 'all': Load all available years (default)
               - Single year: 2000
               - List of years: [2000, 2001, 2002]
               - Range: range(2000, 2024)

    Returns:
        DataFrame with columns including: date, region, extent_mkm2, and ERA5 variables
        (t2m_mean, msl_mean, tp_mean, wind_speed_mean, etc.).

    Raises:
        FileNotFoundError: If ERA5 parquet files don't exist for specified years.
        ValueError: If specified regions don't exist in the data.

    Examples:
        >>> # Load all regions for all years
        >>> df = load_data()

        >>> # Load all regions for specific years
        >>> df = load_data(years=range(2000, 2006))

        >>> # Load specific region for all years
        >>> df = load_data(regions='Central')

        >>> # Load multiple regions for specific years
        >>> df = load_data(regions=['Bering', 'Central'], years=[2000, 2001])
    """
    # Handle "all" for regions
    if regions == 'all':
        regions = ALL_REGIONS
    elif isinstance(regions, str):
        regions = [regions]

    # Handle "all" for years
    if years == 'all':
        years = get_available_years()
    elif isinstance(years, int):
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

    # Interpolate missing ice extent values (handles every-other-day data from 1979-1989)
    df_combined = _interpolate_missing_dates(df_combined)

    return df_combined
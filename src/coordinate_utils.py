"""Utilities for coordinate transformations and geospatial operations on Arctic sea ice data.

This module provides functions for:
- Longitude coordinate transformations between -180/180 and 0/360 conventions
- Loading and processing Arctic region shapefiles
- Slicing xarray datasets to specific geographic regions
"""

from typing import Union

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import xarray as xr


def lon_to_360(x: float) -> float:
    """Convert longitude from -180/180 to 0/360 convention.

    Args:
        x: Longitude in -180/180 format.

    Returns:
        Longitude in 0/360 format.
    """
    return (360 + (x % 360)) % 360


def lon_to_180(lon: float) -> float:
    """Convert longitude from 0/360 to -180/180 convention.

    Args:
        lon: Longitude in 0/360 format.

    Returns:
        Longitude in -180/180 format.
    """
    return ((lon + 180) % 360) - 180

def get_region_shape(
    region: str,
    shapefile: str = '../data/raw/shapefiles_regions/NSIDC-0780_SeaIceRegions_NH_v1.0.shp'
) -> Union[Polygon, MultiPolygon]:
    """Extract the geometry for a specific Arctic region from NSIDC shapefile.

    Args:
        region: Name of the region to extract (e.g., 'Beaufort', 'Barents').
        shapefile: Path to the NSIDC sea ice regions shapefile.

    Returns:
        Shapely geometry representing the region boundary.
    """
    gdf = gpd.read_file(shapefile)
    region_of_interest = gdf[gdf['Region'] == region]
    geometry_roi = region_of_interest['geometry']
    return geometry_roi.iloc[0]


def get_pan_arctic_shape() -> Union[Polygon, MultiPolygon]:
    """Create pan-Arctic geometry by merging all Arctic regions except marginal seas.

    Excludes: Baltic, Japan, Bohai, Gulf of Alaska, St. Lawrence, and Okhotsk seas.
    The geometry is reprojected to EPSG:4326 (WGS84) coordinate system.

    Returns:
        Shapely geometry representing the pan-Arctic region.
    """
    gdf = gpd.read_file('../data/raw/shapefiles_regions/NSIDC-0780_SeaIceRegions_NH_v1.0.shp').to_crs("EPSG:4326")
    name_col = "Region"
    excl = {"Baltic","Japan","Bohai","Gulf_Alaska","St_Lawr","Okhotsk"}
    incl = [n for n in gdf[name_col] if n not in excl]
    print(incl)
    pan_arctic_geom = unary_union(gdf[gdf[name_col].isin(incl)].geometry)
    return pan_arctic_geom


def slice_dataset_to_region(dataset: xr.Dataset, region: str) -> xr.Dataset:
    """Slice an xarray dataset to a specific Arctic region using spatial masking.

    Creates a point-wise mask based on whether each coordinate falls within the
    specified region's geometry. This is particularly useful for filtering ERA5
    or other gridded climate data to specific Arctic regions.

    Args:
        dataset: xarray Dataset with 'longitude' and 'latitude' coordinates.
        region: Region name ('pan_arctic' or specific region like 'Beaufort').

    Returns:
        Filtered dataset containing only points within the specified region.
    """
    if region == "pan_arctic":
        geom = get_pan_arctic_shape()
    else:
        geom = get_region_shape(region)
    lon = dataset["longitude"].compute().values
    lat = dataset["latitude"].compute().values
    pts = shapely.points(lon, lat)
    mask_vals = shapely.contains(geom, pts)
    mask = xr.DataArray(mask_vals, dims=("values",))
    return dataset.where(mask, drop=True)

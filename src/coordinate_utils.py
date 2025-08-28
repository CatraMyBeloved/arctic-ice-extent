import geopandas as gpd
import shapely
from shapely.ops import unary_union
import xarray as xr

def lon_to_360(x):
    return (360 + (x % 360)) % 360

def lon_to_180(lon):
    return ((lon + 180) % 360) - 180

def get_region_shape(region, shapefile = '../data/raw/shapefiles_regions/NSIDC-0780_SeaIceRegions_NH_v1.0.shp'):
    gdf = gpd.read_file(shapefile)

    region_of_interest = gdf[gdf['Region'] == region]

    geometry_roi = region_of_interest['geometry']

    return geometry_roi.iloc[0]


def get_pan_arctic_shape():
    gdf = gpd.read_file('../data/raw/shapefiles_regions/NSIDC-0780_SeaIceRegions_NH_v1.0.shp').to_crs("EPSG:4326")

    name_col = "Region"
    excl = {"Baltic","Japan","Bohai","Gulf_Alaska","St_Lawr","Okhotsk"}
    incl = [n for n in gdf[name_col] if n not in excl]
    print(incl)

    pan_arctic_geom = unary_union(gdf[gdf[name_col].isin(incl)].geometry)
    return pan_arctic_geom

def slice_dataset_to_region(dataset, region):
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

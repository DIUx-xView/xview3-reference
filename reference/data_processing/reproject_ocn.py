"""
Reproject and resample NetCDF4 fields.

1. Convert NetCDF lon/lat/field -> GeoTIFF w/GCPs
2. Reproject GeoTIFF from radar coords to EPSG:4326
3. Resample reprojected OCN to match GRD grid

"""
import sys
from random import sample
from subprocess import run

import netCDF4 as nc
import numpy as np
from osgeo import gdal

variables = ["owiWindSpeed", "owiWindDirection"]


def get_gcp_str(lon, lat, samples=250):
    """2D lon/lat -> "-gcp pixel_x pixel_y lon lat"."""
    gcps = []
    lon = np.flipud(lon)  # important: (0,0) -> upper left
    lat = np.flipud(lat)
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            x, y = lon[i, j], lat[i, j]
            gcps.append(f"-gcp {j} {i} {x} {y}")
    return " ".join(sample(gcps, samples))


# Reads in a pair GRD/OCN
if ".nc" in sys.argv[1]:
    ocnfile, grdfile = sys.argv[1:]
else:
    grdfile, ocnfile = sys.argv[1:]


# Get lat/lon 2d grids from OCN
ds = nc.Dataset(ocnfile)
lon = ds.variables["owiLon"][:]
lat = ds.variables["owiLat"][:]

# Get GRD grid extent
img = gdal.Open(grdfile)
print(img)
width = img.RasterXSize
height = img.RasterYSize

# Convert 2D lat/lon to GCPs
gcps = get_gcp_str(lon, lat)

for var in variables:
    print(f"REPROJECTING {var} ...")

    uri = f'NETCDF:"{ocnfile}":{var}'
    tmpfile = ocnfile.replace(".nc", f"_{var}_tmp.tif")
    outfile = ocnfile.replace(".nc", f"_{var}_interp.tif")

    cmd1 = f"gdal_translate {gcps} -of GTiff {uri} {tmpfile}"
    cmd2 = (
        f"gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 "
        f"-ts {width} {height} -r bilinear -overwrite {tmpfile} {outfile}"
    )
    cmd3 = f"rm {tmpfile}"

    run(cmd1, shell=True)
    run(cmd2, shell=True)
    run(cmd3, shell=True)

"""
Reproject Sentinel-1 GRD images from WGS84 to UTM.

These data processing scripts are for the xView3-SAR dataset
accessible at https://iuu.xview.us/

A complete explanation of what these scripts do is available in
the xView3-SAR paper at https://arxiv.org/abs/2206.00897

"""
import sys
from subprocess import Popen

import numpy as np
from osgeo import gdal

pxlsize = 10
interp = "bilinear"
suffix = "_UTM_10m_16b.tif"


def get_utm_epsg(lon, lat):
    """Return UTM EPSG code of respective lon/lat.

    The EPSG is:
        32600+zone for positive latitudes
        32700+zone for negatives latitudes
    """
    zone = int(round((183 + lon) / 6, 0))
    epsg = int(32700 - round((45 + lat) / 90, 0) * 100) + zone
    print(f"EPSG:{epsg} (UTM {zone})")
    return epsg


def get_center_coord(ds):
    """Return center-pixel coordinates from GDAL raster."""
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx, maxx = gt[0], gt[0] + width * gt[1] + height * gt[2]
    miny, maxy = gt[3] + width * gt[4] + height * gt[5], gt[3]
    return (minx + maxx) / 2, (miny + maxy) / 2


def get_center_point(lons, lats, nodata=-99999):
    """Return center coordinates from lon/lat arrays."""
    lon = np.nanmean(lons[(lons != -99999) & (lons != 0) & ~np.isnan(lons)])
    lat = np.nanmean(lats[(lats != -99999) & (lats != 0) & ~np.isnan(lats)])
    return (lon, lat)


infile = sys.argv[1]

ds = gdal.Open(infile)
lon, lat = get_center_coord(ds)
utm_epsg = get_utm_epsg(lon, lat)

# import rasterio
# src = rasterio.open(Path(infile).parent / 'latitude.tiff')
# img_lat = src.read(1)
# src = rasterio.open(Path(infile).parent / 'longitude.tiff')
# img_lon = src.read(1)
# lon, lat = get_center_point(img_lon, img_lat)
# utm_epsg = get_utm_epsg(lon, lat)

outfile = infile.replace(".tif", suffix)

cmd = (
    f"gdalwarp -s_srs EPSG:4326 -t_srs EPSG:{utm_epsg} "
    f"-tr {pxlsize} {pxlsize} -r {interp} -ot Int16 "
    f"-srcnodata -99999 -dstnodata -32768 {infile} {outfile}"
)
print(f"\n{cmd}\n")
Popen(cmd, shell=True).wait()

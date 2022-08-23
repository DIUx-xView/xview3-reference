#!/usr/bin/python
"""
Sentinel-1 Radiometric Terrain Correction using ESA's SNAP software.

Wrapper around SNAP's gpt CLI and GDAL CLI tools.


ESA STEP Forum
--------------

for SLC data I would suggest

- TOPS Split
- Apply Orbit file
- Thermal Noise Removal
- Calibration to Beta0*
- TOPSAR Deburst
- Radiometric terrain flattening
- (Speckle filtering)
- Range Doppler Terrain Correction

If you don’t need the phase information you can also
download it as a GRD product and only apply the following:

- Apply Orbit file
- Thermal noise removal
- Calibration to Beta0
- Radiometric terrain flattening
- (Speckle filtering)
- Range Doppler Terrain Correction

or (if you don’t have much topography):

- Apply Orbit file
- Thermal noise removal
- Calibration to Sigma0
- (Speckle filtering)
- Range Doppler Terrain Correction

Notes:

You should use UTM only if your area lies completely within one
zone, or just overlaps a bit with the next one. The distortion
can be significant when applying reprojection parameters of one
UTM zone on an area which is in another zone.

Precise orbits (POEORB) take about 20 days from acquisition
to be available. For almost all practical purposes including
InSAR-processing the restituted orbits (RESORB) are good enough.


Google Earth Engine
-------------------

Imagery in the Earth Engine 'COPERNICUS/S1_GRD' Sentinel-1
ImageCollection consists of Level-1 Ground Range Detected (GRD)
scenes processed to backscatter coefficient (σ°) in decibels (dB).
The backscatter coefficient represents target backscattering area
(radar cross-section) per unit ground area. Because it can vary by
several orders of magnitude, it is converted to dB as 10*log10σ°.
It measures whether the radiated terrain scatters the incident
microwave radiation preferentially away from the SAR sensor dB < 0)
or towards the SAR sensor dB > 0). This scattering behavior depends
on the physical characteristics of the terrain, primarily the
geometry of the terrain elements and their electromagnetic
characteristics.

Earth Engine uses the following preprocessing steps
(as implemented by the Sentinel-1 Toolbox) to derive the
backscatter coefficient in each pixel:

1. Apply orbit file
   Updates orbit metadata with a restituted orbit file.

2. GRD border noise removal
   Removes low intensity noise and invalid data on scene edges.
   (As of January 12, 2018)

3. Thermal noise removal
   Removes additive noise in sub-swaths to help reduce
   discontinuities between sub-swaths for scenes in
   multi-swath acquisition modes.
   (This operation cannot be applied to images produced
   before July 2015)

4. Radiometric calibration
   Computes backscatter intensity using sensor calibration
   parameters in the GRD metadata.

5. Terrain correction (orthorectification)
   Converts data from ground range geometry,
   which does not take terrain into account,
   to σ° using the SRTM 30 meter DEM or the ASTER DEM for
   high latitudes (greater than 60° or less than -60°).

Dataset Notes

- Radiometric Terrain Flattening is not being applied
  due to artifacts on mountain slopes.

- The unitless backscatter coefficient is converted
  to dB as described above.

- Sentinel-1 SLC data cannot currently be ingested,
  as Earth Engine does not support images with complex
  values due to inability to average them during
  pyramiding without losing phase information.

- GRD SM assets are not ingested because the
  computeNoiseScalingFactor() function in the border
  noise removal operation in the S1 toolbox does not
  support the SM mode.


AWS RTC Indigo Ag
-----------------

To preprocess the GRD scenes to RTC,
we use the Sentinel-1 Toolbox to:

1. Apply the orbit file
   Refine positioning information. Restituted orbit files are
   used for products added after launch. The more accurate
   precise orbit files are used for historical imagery.

2. Apply thermal noise removal
   Remove thermal noise from acquisition.

3. Apply GRD product border noise removal
   Removes “noisy” pixels along the edges of GRD scenes.
   These pixels are significantly lower than expected but
   not labeled as NODATA in the GRD product.

4. Apply product calibration
   Calibrate the product image “Digital Numbers” (DNs) to
   radiometrically calibrated backscatter.

5. Apply the Lee Sigma speckle filter
   Parameter choices for the Lee Sigma filter are,
   Window size: 7x7; Target window size: 5x5;
   Number of looks: 2; Sigma: 0.9

6. Apply terrain flattening
   We use the SRTM 1Sec HGT resampled using bilinear
   interpolation as the DEM.

7. Apply terrain correction
   We use the same SRTM 1Sec HGT DEM. The image is exported
   in EPSG:4326 with a pixel resolution of ~20m.


Update SNAP (important)
-----------------------

First you have to call:

/path/to/snap --nosplash --nogui --modules --list --refresh

This will check the repository for the latest updates. Then:

/path/to/snap --nosplash --nogui --modules --update-all

This should update your installation.


Further reading
---------------

STEP Forum radiometric correction:
https://forum.step.esa.int/t/radiometric-geometric-correction-workflow/2540

Apply-Orbit-File without internet:
https://forum.step.esa.int/t/apply-orbit-file-step-without-internet/26318

Orbit File timeout connection:
https://forum.step.esa.int/t/orbit-file-timeout-march-2021/28621/54

Orbit Files URL:
http://step.esa.int/auxdata/orbits/Sentinel-1/

DEM Files URL:
http://step.esa.int/auxdata/dem/SRTMGL1/

Terrain-Correction projection definition:
https://forum.step.esa.int/t/snap-gpt-terrain-correction/498/3

"""
import sys
from pathlib import Path
from subprocess import Popen
from datetime import datetime

global gpt
global outdir
global pxlsize
global cleanup

# --- EDIT ----------------------------------------

# Point to the SNAP gpt executable on your system.
# gpt = "/opt/snap/bin/gpt"
gpt = "/Applications/snap/bin/gpt"

# Path for output resutls and temporary files.
outdir = "data/galapagos/processed"

# Output product resolution
# (S1 GRD is gridded at 10 m, native resolution is 20 m)
pxlsize = 20.0

# Apply speckle filter
depeckle = False

# Remove all temporary data files
cleanup = False

# ------------------------------------------------


def run_operator(
    gpt=gpt,
    operator=None,
    params=None,
    source=None,
    outdir=None,
    suffix=None,
):
    """Run SNAP's gpt command-line tool.

    gpt: '/path/to/snap/bin/gpt'
    operator: 'Calibration' | 'Terrain-Correction' | ..
    params: ['-Pparam1=option1', '-Pparam2=option2', ..]
    source: file.zip or file.dim (XML)
    outdir: '/path/to/output/dir'
    suffix: 'ID' (appended to output files)
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    target = Path(outdir) / (Path(source).stem + f"_{suffix}.dim")
    cmd = (
        f"{gpt} "
        f"{operator} -e "
        f"{' '.join(params)} "
        f"-Ssource={source} "
        f"-SsourceProduct={source} "  # different for some operators
        f"-t {target} "
    )
    print(f"\nRUNNING: {operator} ...")
    print(cmd)
    Popen(cmd, shell=True).wait()
    return target


def log_transform(ftif, suffix="LOG"):
    """Log scaling (as per SNAP/Google) [linear to dB scale]:

       L = 10 * log10(sig0)
    """
    fout = str(ftif).replace(".tif", f"_{suffix}.tif")
    calc = "'10 * log10(A + 0.00001)'"
    cmd = (
        f"gdal_calc.py -A {ftif} --outfile={fout} --overwrite "
        f"--calc={calc} --NoDataValue=0 --quiet"
    )
    Popen(cmd, shell=True).wait()
    return fout


def qpower_transform(ftif, suffix="QPWR"):
    """Quarter-Power scaling:

        P = 255 * B * sqrt(pxl)
        B = 1 / [3 * median(sqrt(pxl))]
    """
    fout = str(ftif).replace(".tif", f"_{suffix}.tif")
    calc = "'255 * (1 / (3 * median(sqrt(A)))) * sqrt(A)'"
    cmd = (
        f"gdal_calc.py -A {ftif} --outfile={fout} --overwrite "
        f"--calc={calc} --NoDataValue=0 --quiet"
    )
    Popen(cmd, shell=True).wait()
    return fout


def get_subdir(outdir, source):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    return Path(outdir) / Path(source).stem


def list_files(path, glob="*.img"):
    """List files in `path` recursively."""
    return [str(p) for p in Path(path).rglob(glob)]


def remove_content(path, skip=".tif"):
    fs = [str(o) for o in Path(path).glob("*") if skip not in str(o)]
    cmd = f"rm -rfv {' '.join(fs)}"
    Popen(cmd, shell=True).wait()


def img_to_geotiff(fimg):
    """Convert ENVI files to GeoTIFF."""
    ftif = str(fimg).replace(".img", ".tif")
    cmd = f"gdal_translate -of GTiff {fimg} {ftif}"
    Popen(cmd, shell=True).wait()
    return ftif


def geotiff_to_jpeg(ftif, vmin=-35, vmax=-5):
    """Convert GeoTIFF to scaled JPEG."""
    fjpg = str(ftif).replace(".tif", ".jpg")
    cmd = (
        f"gdal_translate -of JPEG -scale {vmin} {vmax} "
        f"-ot Byte -co 'QUALITY=100' -co worldfile=yes {ftif} {fjpg}"
    )
    Popen(cmd, shell=True).wait()
    return fjpg


def move_geotiff(ftif, outdir):
    """Rename geotiff files and move a folder up."""
    name1 = Path(outdir).name
    name2 = Path(ftif).name
    ftif2 = Path(outdir) / f"{name1}_{name2}"
    cmd = f"mv {ftif} {ftif2}"
    Popen(cmd, shell=True).wait()
    return ftif2


def print_time(time1, time2):
    print("Time:", time2 - time1)
    return time2


# ------------------------------------------------


def exec_preprocessing(fin):

    start_time = datetime.now()

    subdir = get_subdir(outdir, fin)

    fout = run_operator(
        operator="Apply-Orbit-File",
        params=[
            "-PcontinueOnFail=false",
            "-PorbitType='Sentinel Precise (Auto Download)'",
            # "-PorbitType='Sentinel Restituted (Auto Download)'",
        ],
        source=fin,
        outdir=subdir,
        suffix="ORB",
    )
    last_time = print_time(start_time, datetime.now())

    fout = run_operator(
        operator="Remove-GRD-Border-Noise",
        params=["-PborderLimit=500", "-PtrimThreshold=0.5"],
        source=fout,
        outdir=subdir,
        suffix="BNR",
    )
    last_time = print_time(last_time, datetime.now())

    fout = run_operator(
        operator="ThermalNoiseRemoval",
        params=["-PremoveThermalNoise=true"],
        source=fout,
        outdir=subdir,
        suffix="TNR",
    )
    last_time = print_time(last_time, datetime.now())

    fout = run_operator(
        operator="Calibration",
        params=["-PoutputBetaBand=false", "-PoutputSigmaBand=true"],
        source=fout,
        outdir=subdir,
        suffix="CAL",
    )
    last_time = print_time(last_time, datetime.now())

    if depeckle:
        fout = run_operator(
            operator="Speckle-Filter",
            params=[
                "-Pfilter='Refined Lee'",
                "-PwindowSize='7x7'",
                "-PtargetWindowSizeStr='5x5'",
            ],
            source=fout,
            outdir=subdir,
            suffix="LEE",
        )
        last_time = print_time(last_time, datetime.now())

    fout = run_operator(
        operator="Terrain-Correction",
        params=[
            "-PsaveDEM=true",
            "-PsaveLatLon=true",
            "-PsaveSigmaNought=true",
            "-PsaveProjectedLocalIncidenceAngle=true",
            f"-PpixelSpacingInMeter={pxlsize}",
            # "-PmapProjection=WGS84",  # default WGS84  # TODO: Try full WKT
            "-PdemName='SRTM 1Sec HGT'",
            "-PnodataValueAtSea=false",
        ],
        source=fout,
        outdir=subdir,
        suffix="TC",
    )
    last_time = print_time(last_time, datetime.now())

    # List all .img files from last .data output
    data_out = str(fout).replace(".dim", ".data")
    files = list_files(data_out)

    for fimg in files:
        ftif = img_to_geotiff(fimg)
        ftif = move_geotiff(ftif, subdir)
        if "Sigma0" in ftif.name:
            ftif = log_transform(ftif)
            fjpg = geotiff_to_jpeg(ftif)
            print("->", fjpg)

    print("\nTotal processing time:")
    last_time = print_time(start_time, datetime.now())

    return subdir


if __name__ == "__main__":

    files = sys.argv[1:]

    if not files:
        print("Usage: process_grd.py {files.zip | files.dim}")

    for f in files:
        subdir = exec_preprocessing(f)

    if cleanup:
        # FIXME: This is not robust
        remove_content(subdir, skip=".tif")

#!/usr/bin/env python

"""
Process downloaded MOD09A1 surface reflectance HDF files into clipped, reprojected RGB reflectance files 
for use as a MODIS "true color" background in map-view figures.
"""

import os
import re
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

# --- USER SETTINGS ---
SR_DIR    = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/MODIS_SR"
AOI_FILE  = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling/MCS_outline/basin_outline.shp"
OUT_DIR   = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/MODIS_SR/clipped"
OVERWRITE = False
# ---------------------

# Band names for a true-color RGB (500 m surface reflectance bands)
RGB_BANDS = {"red": "sur_refl_b01", "green": "sur_refl_b04", "blue": "sur_refl_b03"}
SCALE_FACTOR = 0.0001 


def file_date(hdf_file):
    """
    Start-of-composite date from a MOD09A1 filename
    """
    match = re.search(r"\.A(\d{4})(\d{3})\.", os.path.basename(hdf_file))
    if not match:
        raise ValueError(f"No AYYYYDDD date found in {os.path.basename(hdf_file)}")
    year, doy = int(match.group(1)), int(match.group(2))
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1)


def clipped_path(hdf_file, out_dir):
    """
    Path of the reprojected, clipped RGB file for a MOD09A1 HDF.
    """
    return os.path.join(out_dir, f"MOD09A1_RGB_{file_date(hdf_file):%Y%m%d}_clip.tif")


def clip_modis_sr_to_aoi(hdf_file, aoi, out_dir, overwrite=False):
    """
    Reproject a MOD09A1 HDF's red/green/blue bands to the AOI's CRS and clip to the AOI geometry,
    keeping the native ~463 m sinusoidal resolution. Saves a 3-band GeoTIFF (red, green, blue) to
    out_dir and returns its path.
    """
    out_file = clipped_path(hdf_file, out_dir)
    if os.path.exists(out_file) and not overwrite:
        return out_file

    ds = rxr.open_rasterio(hdf_file, variable=list(RGB_BANDS.values()), masked=True).squeeze()

    # Crop to the AOI bounds in the native sinusoidal projection first (much faster), then
    # reproject and clip. 
    native_res = abs(ds.rio.resolution()[0])
    xmin, ymin, xmax, ymax = aoi.to_crs(ds.rio.crs).total_bounds
    ds_crop = ds.rio.clip_box(
        xmin - 2 * native_res, ymin - 2 * native_res, xmax + 2 * native_res, ymax + 2 * native_res
        )
    ds_clip = ds_crop.rio.reproject(aoi.crs, resolution=native_res).rio.clip(aoi.geometry)

    rgb = xr.concat(
        [ds_clip[RGB_BANDS[c]] * SCALE_FACTOR for c in ("red", "green", "blue")],
        dim=pd.Index([1, 2, 3], name="band"),
        )
    rgb = rgb.rio.write_crs(ds_clip.rio.crs).rio.write_nodata(np.nan)
    rgb.astype("float32").rio.to_raster(out_file)
    return out_file


def process_modis_sr(sr_dir, aoi_file, out_dir, overwrite=False):
    """
    Clip every MOD09A1 HDF in sr_dir to the AOI and save an RGB GeoTIFF per date in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    aoi = gpd.read_file(aoi_file)
    files = sorted(glob(os.path.join(sr_dir, "MOD09A1*.hdf")))
    if not files:
        raise FileNotFoundError(f"No MOD09A1 HDF files found in {sr_dir}")

    for f in tqdm(files, desc="Clipping MOD09A1 RGB to the AOI"):
        clip_modis_sr_to_aoi(f, aoi, out_dir, overwrite=overwrite)
    print(f"Clipped MODIS RGB files saved to {out_dir}")


def main():
    process_modis_sr(SR_DIR, AOI_FILE, OUT_DIR, overwrite=OVERWRITE)


if __name__ == "__main__":
    main()

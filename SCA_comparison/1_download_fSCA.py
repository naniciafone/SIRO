#!/usr/bin/env python

"""
Download SPIReS-MODIS/Terra fSCA files from the Snow Today FTP server for a time span and area of interest.

The MODIS tiles are determined from the AOI file. 
MODIS tiles form a fixed 36 x 18 grid of ~1,112 km squares in the MODIS sinusoidal projection.

Adapted from the NSIDC example:
https://nsidc.org/data/user-resources/help-center/how-access-nsidc-data-using-ftp-client-command-line-wget-or-python
"""

import os
import re
from ftplib import FTP, error_perm
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm
import xarray as xr

# --- USER SETTINGS ---
OUT_DIR             = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/SPIReS"
AOI_FILE            = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling/MCS_outline/basin_outline.shp"
DATE_START          = "2022-10-01" # inclusive
DATE_END            = "2025-06-30" # inclusive
MONTHS              = [10, 11, 12, 1, 2, 3, 4, 5, 6] # inclusive
CLIP_TO_AOI         = True
REMOVE_UNCLIPPED    = True
OVERWRITE           = False
DRY_RUN             = False
# ---------------------

FTP_HOST = "dtn.rc.colorado.edu"
DATA_ROOT = "/shares/snow-today/gridded_data/SPIRES_HIST_V01"

# MODIS sinusoidal grid (spherical Earth, R = 6,371,007.181 m)
MODIS_SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
TILE_SIZE = 1111950.5197665233   # tile width and height [m]
X_MIN, Y_MAX = -20015109.355798, 10007554.677899  # upper-left corner of tile h00v00 [m]


def tiles_from_aoi(aoi_file):
    """
    Identify MODIS sinusoidal tiles that intersect the AOI geometry.
    """
    aoi = gpd.read_file(aoi_file).to_crs(MODIS_SINUSOIDAL).union_all()
    xmin, ymin, xmax, ymax = aoi.bounds
    h_range = range(int((xmin - X_MIN) // TILE_SIZE), int((xmax - X_MIN) // TILE_SIZE) + 1)
    v_range = range(int((Y_MAX - ymax) // TILE_SIZE), int((Y_MAX - ymin) // TILE_SIZE) + 1)

    tiles = []
    for h in h_range:
        for v in v_range:
            tile = box(
                X_MIN + h * TILE_SIZE, Y_MAX - (v + 1) * TILE_SIZE,
                X_MIN + (h + 1) * TILE_SIZE, Y_MAX - v * TILE_SIZE
                )
            if tile.intersects(aoi):
                tiles.append(f"h{h:02d}v{v:02d}")
    return tiles


def file_date(filename):
    """
    Date of a SPIReS file, from the first YYYYMMDD in its name.
    """
    match = re.search(r"(\d{8})", filename)
    return pd.to_datetime(match.group(1), format="%Y%m%d") if match else None


def clipped_path(image_file):
    """Path of the reprojected, clipped version of a downloaded SPIReS file."""
    return os.path.splitext(image_file)[0] + "_clip.nc"


def clip_image_to_aoi(image_file, aoi, all_touched=False):
    """
    Reproject a SPIReS file to the AOI's CRS and clip it to the AOI geometry. Saves all variables
    to <image_file>_clip.nc and returns that path. 
    """
    out_file = clipped_path(image_file)
    with xr.open_dataset(image_file, decode_coords="all", mask_and_scale=True) as ds:
        # Crop to the AOI bounds in the native projection first (much faster), then reproject and clip.
        # Keep the native ~463 m resolution. Otherwise, rio.reproject coarsens it to cover the sinusoidal pixels.
        native_res = abs(ds.rio.resolution()[0])
        xmin, ymin, xmax, ymax = aoi.to_crs(ds.rio.crs).total_bounds
        ds_crop = ds.rio.clip_box(xmin - 2 * native_res, ymin - 2 * native_res, xmax + 2 * native_res, ymax + 2 * native_res)
        ds_clip = ds_crop.rio.reproject(aoi.crs, resolution=native_res).rio.clip(aoi.geometry, all_touched=all_touched)
        ds_clip.attrs.update(ds.attrs)

        # Keep each variable's original packing (e.g., uint8 with fill value 255) and CRS link, and compress
        keep = ("dtype", "_FillValue", "scale_factor", "add_offset", "grid_mapping")
        encoding = {
            v: {**{k: e for k, e in ds_clip[v].encoding.items() if k in keep}, "zlib": True, "complevel": 4}
            for v in ds_clip.data_vars
        }
        ds_clip.load().to_netcdf(out_file, encoding=encoding)
    return out_file


def download_spires(
        out_dir, start_date, end_date, aoi_file=None, 
        months=(10, 11, 12, 1, 2, 3, 4, 5, 6), dry_run=False, overwrite=False,
        clip_to_aoi=False, remove_unclipped=True
        ):
    """
    Download SPIReS fSCA files for the MODIS tiles covering aoi_file (or the given tiles)
    between start_date and end_date (inclusive), keeping only the given months.
    """
    # Check dates and AOI
    start, end = pd.to_datetime(start_date), pd.to_datetime(end_date)
    os.makedirs(out_dir, exist_ok=True)
    if clip_to_aoi and aoi_file is None:
        raise ValueError("clip_to_aoi requires aoi_file.")
    aoi = gpd.read_file(aoi_file) if clip_to_aoi else None

    # Get tiles to download from AOI
    tiles = tiles_from_aoi(aoi_file)
    print(f"MODIS tiles: {', '.join(tiles)}")
    print(f"Dates: {start:%Y-%m-%d} to {end:%Y-%m-%d}, months {list(months)}")

    # Connect and log in to the FTP
    ftp = FTP(FTP_HOST)
    ftp.login("anonymous")

    # Files are organized by tile, then calendar year
    for tile in tiles:
        for year in range(start.year, end.year + 1):
            data_dir = f"{DATA_ROOT}/{tile}/{year}"
            try:
                ftp.cwd(data_dir)
            except error_perm:
                print(f"\n{data_dir} not found on the server, skipping.")
                continue

            # Keep files within the date range and months
            files = []
            for file in ftp.nlst():
                date = file_date(file)
                if date is not None and start <= date <= end and date.month in months:
                    files.append(file)

            # Decide what still needs to be done for each file (e.g., whether expected outputs already exist)
            to_download, to_clip = [], []
            for file in files:
                raw_file = os.path.join(out_dir, file)
                done = os.path.exists(clipped_path(raw_file) if clip_to_aoi else raw_file)
                if done and not overwrite:
                    continue
                if overwrite or not os.path.exists(raw_file):
                    to_download.append(file)
                if clip_to_aoi:
                    to_clip.append(file)
            message = f"\n{data_dir}: {len(files)} files in range, {len(to_download)} to download"
            if clip_to_aoi:
                message += f", {len(to_clip)} to clip"
            print(message)
            if dry_run or not (to_download or to_clip):
                continue

            for file in tqdm(sorted(set(to_download) | set(to_clip)), desc=f"{tile} {year}"):
                raw_file = os.path.join(out_dir, file)
                if file in to_download:
                    try:
                        with open(raw_file, "wb") as f:
                            ftp.retrbinary("RETR " + file, f.write)
                    except Exception:
                        # Remove partial downloads so they are retried next time
                        if os.path.exists(raw_file):
                            os.remove(raw_file)
                        raise
                if file in to_clip:
                    clip_image_to_aoi(raw_file, aoi)
                    if remove_unclipped:
                        os.remove(raw_file)

    ftp.quit()


def main():
    download_spires(
        OUT_DIR, DATE_START, DATE_END, aoi_file=AOI_FILE,
        months=MONTHS, dry_run=DRY_RUN, overwrite=OVERWRITE,
        clip_to_aoi=CLIP_TO_AOI, remove_unclipped=REMOVE_UNCLIPPED
        )


if __name__ == "__main__":
    main()

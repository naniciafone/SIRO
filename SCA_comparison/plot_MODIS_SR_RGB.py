#!/usr/bin/env python

"""
Save a quick-look PNG for each MOD09GA file produced by download_MODIS_SR.py, for manual cloud screening.

Each PNG shows a true-color (bands 1-4-3) panel and, optionally, a false-color (bands 7-2-1) panel,
where snow appears cyan/blue and clouds appear white, which makes them easier to tell apart.
A review CSV (date, file, blank "clear" column) is also written for recording your decisions.
"""

import os
import re
from glob import glob

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from tqdm import tqdm

# --- USER SETTINGS ---
IN_DIR            = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/MODIS_SR_daily/clipped"
OUT_DIR           = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/MODIS_SR_daily/quicklooks"
AOI_FILE          = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling/MCS_outline/basin_outline.shp" # None = no outline
PLOT_FALSE_COLOR  = True # add a bands 7-2-1 panel next to the true-color panel
REFL_MAX          = 1.0 # reflectance mapped to full brightness (same for every image so dates are comparable)
GAMMA             = 0.7 # < 1 brightens darker (snow-free) terrain
OVERWRITE         = False
# ---------------------

TRUE_COLOR = ("sur_refl_b01_1", "sur_refl_b04_1", "sur_refl_b03_1")
FALSE_COLOR = ("sur_refl_b07_1", "sur_refl_b02_1", "sur_refl_b01_1")


def file_date(nc_file):
    """
    Date from a download_MODIS_SR.py output name (YYYYMMDD in clipped files, AYYYYDDD in reprojected files).
    """
    name = os.path.basename(nc_file)
    match = re.search(r"_(\d{8})_", name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    match = re.search(r"\.A(\d{4})(\d{3})\.", name)
    return pd.Timestamp(int(match.group(1)), 1, 1) + pd.Timedelta(days=int(match.group(2)) - 1)


def make_rgb(ds, bands, refl_max=1.0, gamma=0.7):
    """
    Stack three reflectance bands into an (y, x, 3) array scaled to 0-1, with NaNs shown as white.
    """
    rgb = np.dstack([ds[b].values for b in bands])
    rgb = np.clip(rgb / refl_max, 0, 1) ** gamma
    nodata = np.isnan(rgb).any(axis=2)
    rgb[nodata] = 1.0
    return rgb


def plot_file(nc_file, out_file, aoi=None, false_color=True, refl_max=1.0, gamma=0.7):
    """
    Plot true-color (and optionally false-color) panels for one file and save as PNG.
    """
    with xr.open_dataset(nc_file, decode_coords="all") as ds:  # applies scale_factor and fill values
        ds = ds.squeeze(drop=True).load()
    extent = [float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())]

    panels = [("True color (1-4-3)", TRUE_COLOR)]
    if false_color:
        panels.append(("False color (7-2-1): snow cyan, cloud white", FALSE_COLOR))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    for ax, (title, bands) in zip(axes[0], panels):
        ax.imshow(make_rgb(ds, bands, refl_max, gamma), extent=extent, origin="upper", interpolation="nearest")
        if aoi is not None:
            aoi.to_crs(ds.rio.crs).boundary.plot(ax=ax, color="red", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{file_date(nc_file):%Y-%m-%d}  ({os.path.basename(nc_file)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    aoi = gpd.read_file(AOI_FILE) if AOI_FILE else None
    files = sorted(glob(os.path.join(IN_DIR, "*.nc")), key=file_date)
    if not files:
        raise FileNotFoundError(f"No .nc files found in {IN_DIR}")

    for f in tqdm(files, desc="Plotting RGB quick-looks"):
        out_file = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(f))[0] + ".png")
        if os.path.exists(out_file) and not OVERWRITE:
            continue
        try:
            plot_file(
                f, out_file, aoi=aoi, false_color=PLOT_FALSE_COLOR, refl_max=REFL_MAX, gamma=GAMMA
                )
        except Exception as err:
            tqdm.write(f"{os.path.basename(f)} failed: {err}")

    # Review sheet for manual cloud screening; never overwrite one you've started filling in
    review_csv = os.path.join(OUT_DIR, "cloud_review.csv")
    if not os.path.exists(review_csv):
        pd.DataFrame(
            {"date": [file_date(f).date() for f in files], "file": [os.path.basename(f) for f in files], "clear": ""}
            ).to_csv(review_csv, index=False)
        print(f"Review sheet written to {review_csv}")
    print(f"Quick-looks saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

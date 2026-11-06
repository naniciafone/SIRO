#!/usr/bin/env python

import glob
import os
import re
import warnings
from datetime import datetime
import fiona
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject

from model_comparison_functions import LIDAR_NAMES, MODEL_NAMES

# ----- Settings -----
# Path to "dates"
DATA_DIR = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling/dates"

TASKS = [1,2]
NODATA = -9999
INCHES_TO_M = 0.0254

# Raw model snow depth files in modeled/Task<N>/: glob pattern, multiplier to meters, LiDAR grid
MODELS = {
    "HMS_EB":    {"pattern": "*eb_snow*.tiff",     "to_m": INCHES_TO_M, "lidar": "LiDAR"},
    "HMS_TI":    {"pattern": "*TI_snow*.tif",      "to_m": INCHES_TO_M, "lidar": "LiDAR"},
    "iSnobal":   {"pattern": "*mores_creek*.tif",  "to_m": 1.0,         "lidar": "LiDAR"},
    "SnowModel": {"pattern": "*snod*.tif",         "to_m": 1.0,         "lidar": "LiDAR"},
}
# LiDAR snow depth files in lidar/
LIDAR = {"LiDAR": "*SD.tif", "LiDAR_2000": "*2000*.tif"}


# ----- Helper functions -----
def find_one(folder, pattern):
    matches = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 file matching '{pattern}' in {folder}, "
                                f"found {len(matches)}: {[os.path.basename(m) for m in matches]}")
    return matches[0]

def file_date(path):
    """
    First YYYYMMDD date in a filename
    """
    name = os.path.basename(path)
    match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name) or re.search(r"(\d{4})(\d{2})(\d{2})", name)
    return "".join(match.groups()) if match else None

def read_shapes(vector_file):
    with fiona.open(vector_file, "r") as shapefile:
        return [feature["geometry"] for feature in shapefile]

def read_raster(path, scale=1.0, default_crs=None):
    """
    Read band 1 as float32 in meters, with nodata and NaN set to NODATA.
    """
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float32").filled(np.nan) * np.float32(scale)
        profile = src.profile.copy()
    if profile["crs"] is None:
        warnings.warn(f"{os.path.basename(path)} has no CRS. Assuming {default_crs}.")
        profile["crs"] = default_crs
    data[~np.isfinite(data)] = NODATA
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA, compress="lzw")
    return data, profile

def write_raster(path, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)

def clip(data, profile, shapes, all_touched):
    """
    Clip and crop a raster to shapes. Pixels outside the shapes are set to NODATA.
    """
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        with mem.open() as src:
            out, transform = rasterio.mask.mask(src, shapes, crop=True, all_touched=all_touched, nodata=NODATA)
    profile = profile.copy()
    profile.update(height=out.shape[1], width=out.shape[2], transform=transform)
    return out[0], profile

def resample_to_grid(data, profile, dst_profile):
    """
    Nearest-neighbour resample of a model raster onto another grid.
    """
    out = np.full((dst_profile["height"], dst_profile["width"]), NODATA, dtype="float32")
    reproject(
        source=data, destination=out,
        src_transform=profile["transform"], src_crs=profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.nearest,
    )
    return out


# ----- STATS AND FIGURES -----
def raster_stats(name, data):
    valid = data[data != NODATA]
    stats = {"model": name, "n_valid": valid.size}
    if valid.size:
        stats.update({"min": valid.min(), "mean": valid.mean(), "max": valid.max(), "zeros": int((valid == 0).sum())})
    return stats

def plot_maps(rasters, outline, title, fig_file, ncols):
    """
    Map of each raster (dict of name: (data, profile)) with an outline drawn on top.
    """
    nrows = int(np.ceil(len(rasters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, (name, (data, profile)) in zip(axes, rasters.items()):
        bounds = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
        im = ax.imshow(np.ma.masked_equal(data, NODATA), cmap="viridis", vmin=0, vmax=4.0,
                       extent=[bounds[0], bounds[2], bounds[1], bounds[3]])
        outline.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(rasters):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Snow Depth (m)")
    fig.suptitle(title, fontsize=14)
    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_boxplot(rasters, title, fig_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(rasters)
    ax.boxplot([data[data != NODATA] for data, _ in rasters.values()])
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylabel("Snow Depth (m)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----- MAIN WORKFLOW -----
def prep_date(date_dir, tasks):
    date_str = os.path.basename(date_dir)
    date_label = datetime.strptime(date_str, "%Y%m%d").strftime("%B %d, %Y")
    lidar_domain_file = os.path.join(date_dir, "MCS_outline", "MCS_outline.shp")
    basin_file = os.path.join(date_dir, "MCS_outline", "basin_outline.shp")
    lidar_domain_shapes = read_shapes(lidar_domain_file)
    basin_shapes = read_shapes(basin_file)

    # LiDAR: clip to the LiDAR domain
    lidar = {}
    for tag, pattern in LIDAR.items():
        path = find_one(os.path.join(date_dir, "lidar"), pattern)
        if file_date(path) != date_str:
            raise ValueError(f"{date_str}: LiDAR filename date does not match the folder: {os.path.basename(path)}")
        lidar[tag] = clip(*read_raster(path), lidar_domain_shapes, all_touched=False)
    crs = lidar["LiDAR"][1]["crs"]

    for task in tasks:
        print(f"{date_str}, Task {task}")
        model_dir = os.path.join(date_dir, "modeled", f"Task{task}")
        out_dir = os.path.join(date_dir, "outputs", f"task{task}")
        rasters_dir, figs_dir = os.path.join(out_dir, "rasters"), os.path.join(out_dir, "figs")
        os.makedirs(rasters_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)

        for tag, (data, profile) in lidar.items():
            write_raster(os.path.join(rasters_dir, f"{tag}_MCS_clip.tif"), data, profile)

        basin_clips, lidar_domain_clips = {}, {LIDAR_NAMES[tag]: lidar[tag] for tag in lidar}
        for tag, spec in MODELS.items():
            path = find_one(model_dir, spec["pattern"])
            if file_date(path) != date_str:
                raise ValueError(
                    f"{date_str} Task {task} {tag}: model filename date does not match "
                    f"the LiDAR date: {os.path.basename(path)}"
                    )
            data, profile = read_raster(path, spec["to_m"], default_crs=crs)
            name = MODEL_NAMES[tag]
            if spec["to_m"] != 1.0:
                # Keep a copy of the full model raster after conversion to meters
                write_raster(os.path.join(rasters_dir, f"{tag}_meters.tif"), data, profile)

            # Basin and LiDAR-domain clips, each made separately from the full model raster
            basin_clips[name] = clip(data, profile, basin_shapes, all_touched=True)
            write_raster(os.path.join(rasters_dir, f"{tag}_basin_clip.tif"), *basin_clips[name])
            lidar_domain_clips[name] = clip(data, profile, lidar_domain_shapes, all_touched=False)
            write_raster(os.path.join(rasters_dir, f"{tag}_MCS_clip.tif"), *lidar_domain_clips[name])

            # Model on the LiDAR grid, and model - LiDAR. Resampled from the LiDAR-domain clip, so a
            # LiDAR pixel is only compared if its matched model cell's centre is inside the domain too.
            lidar_data, lidar_profile = lidar[spec["lidar"]]
            resampled = resample_to_grid(*lidar_domain_clips[name], lidar_profile)
            invalid = (resampled == NODATA) | (lidar_data == NODATA)
            resampled[invalid] = NODATA
            diff = np.where(invalid, NODATA, resampled - lidar_data)
            write_raster(os.path.join(rasters_dir, f"{tag}_lidar_resample.tif"), resampled, lidar_profile)
            write_raster(os.path.join(rasters_dir, f"{tag}_lidar_diff.tif"), diff, lidar_profile)

        for label, clips in [("basin", basin_clips), ("MCS", lidar_domain_clips)]:
            pd.DataFrame([raster_stats(name, data) for name, (data, _) in clips.items()]).to_csv(
                os.path.join(figs_dir, f"{label}_stats.csv"), index=False)

        # Make figures
        lidar_outline = gpd.read_file(lidar_domain_file)
        plot_maps(basin_clips, lidar_outline, f"Mores Creek Basin Snow Depth, {date_label}",
                    os.path.join(figs_dir, "Basin_models.png"), ncols=2)
        plot_boxplot(basin_clips, f"Mores Creek Snow Depth, {date_label}",
                        os.path.join(figs_dir, "basin_boxplot.png"))
        plot_maps(lidar_domain_clips, lidar_outline, f"Mores Creek Summit Snow Depth, {date_label}",
                    os.path.join(figs_dir, "MCS_models.png"), ncols=3)
        plot_boxplot(lidar_domain_clips, f"Mores Creek Summit Snow Depth, {date_label}",
                        os.path.join(figs_dir, "MCS_boxplot.png"))


def main():
    date_dirs = sorted(
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if re.fullmatch(r"\d{8}", d) and os.path.isdir(os.path.join(DATA_DIR, d))
        )

    if not date_dirs:
        raise FileNotFoundError(f"No YYYYMMDD folders found in {DATA_DIR}")
    print(f"Processing {len(date_dirs)} dates, tasks {TASKS}")

    for date_dir in date_dirs:
        prep_date(date_dir, TASKS)
    print("Done.")


if __name__ == "__main__":
    main()

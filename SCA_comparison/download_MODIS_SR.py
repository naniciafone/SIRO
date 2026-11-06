#!/usr/bin/env python

"""
Download MODIS/Terra daily surface reflectance (MOD09GA v061, CMR concept ID C2202497474-LPCLOUD)
over an area of interest, reproject to a user-specified CRS, and optionally clip to the AOI.

Adapted from download_earthdata.py. Granules are searched and downloaded with earthaccess
(LP DAAC cloud), then reprojected/clipped locally with rioxarray. MOD09GA is distributed as HDF4,
which Harmony's reprojection/subsetting services don't support for this collection.

Outputs
  - CLIP_TO_AOI = True:  one NetCDF per date in <OUT_DIR>/clipped, with all intersecting tiles
                         mosaicked, reprojected to TARGET_CRS, and clipped to the AOI.
  - CLIP_TO_AOI = False: one NetCDF per granule (tile) in <OUT_DIR>/reprojected, reprojected to TARGET_CRS.
  Raw HDF files are kept in <OUT_DIR>/raw (deleted after clipping if REMOVE_UNCLIPPED = True).
  Re-running skips any date whose processed output exists, and reuses raw HDFs already on disk.

Credentials (earthaccess.login tries these in order):
  - EARTHDATA_USERNAME / EARTHDATA_PASSWORD (or EARTHDATA_TOKEN) environment variables
  - ~/.netrc with
        machine urs.earthdata.nasa.gov login <user> password <pass>
    (chmod 0600 ~/.netrc)
  - Interactive prompt

Requires: earthaccess, geopandas, rioxarray, xarray, netcdf4, tqdm, and GDAL with the HDF4 driver.
"""

import os
import re
from collections import defaultdict

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling
from rioxarray.merge import merge_datasets
from shapely.geometry import box
from tqdm import tqdm

# --- USER SETTINGS ---
OUT_DIR             = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCS_SCA/MODIS_SR_daily"
AOI_FILE            = "/Users/rdcrlrka/Research/SIRO/SIRO_P1/MCSModeling/MCS_outline/basin_outline.shp"
DATE_START          = "2022-10-01" # inclusive
DATE_END            = "2025-06-30" # inclusive
MONTHS              = [10, 11, 12, 1, 2, 3, 4, 5] # inclusive
TARGET_CRS          = "EPSG:32611" # e.g., "EPSG:32611" (UTM 11N); None = use the AOI's CRS
TARGET_RES          = None # output pixel size in TARGET_CRS units; None = native 500 m grid (~463 m)
VARIABLES           = [ # MOD09GA HDF layer names; 1 km layers are resampled (nearest) onto the 500 m grid
    "sur_refl_b01_1", "sur_refl_b02_1", "sur_refl_b03_1", "sur_refl_b04_1",
    "sur_refl_b05_1", "sur_refl_b06_1", "sur_refl_b07_1",
    "QC_500m_1", "state_1km_1",
]
RESAMPLING          = "nearest" # for sur_refl bands ("nearest", "bilinear", "cubic", ...); QA/other layers always use nearest
CLIP_TO_AOI         = True
ALL_TOUCHED         = False # include every pixel touched by the AOI when clipping
REMOVE_UNCLIPPED    = True # delete raw HDFs after a successful clip (only used if CLIP_TO_AOI)
OVERWRITE           = False # re-process (and re-download if needed) even if outputs exist
DRY_RUN             = False # search and report what would be downloaded, without downloading
THREADS             = 10 # parallel downloads
# ---------------------

CONCEPT_ID = "C2202497474-LPCLOUD" # MOD09GA v061

# MODIS sinusoidal grid (spherical Earth, R = 6,371,007.181 m)
MODIS_SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
TILE_SIZE = 1111950.5197665233   # tile width and height [m]
X_MIN, Y_MAX = -20015109.355798, 10007554.677899  # upper-left corner of tile h00v00 [m]


def tiles_from_aoi(aoi):
    """
    Identify MODIS sinusoidal tiles that intersect the AOI geometry.
    """
    aoi = aoi.to_crs(MODIS_SINUSOIDAL).union_all()
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


def parse_granule_name(hdf_name):
    """
    Product, date, and tile from a MODIS filename, e.g. MOD09GA.A2023001.h09v04.061.2023003024802.hdf
    """
    match = re.match(r"(\w+)\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.", os.path.basename(hdf_name))
    if not match:
        raise ValueError(f"Unexpected MODIS filename: {hdf_name}")
    product, year, doy, tile = match.groups()
    date = pd.Timestamp(int(year), 1, 1) + pd.Timedelta(days=int(doy) - 1)
    return product, date, tile


def hdf_url(granule):
    """
    HTTPS link to a granule's HDF file.
    """
    links = [link for link in granule.data_links(access="external") if link.endswith(".hdf")]
    if not links:
        raise ValueError(f"No HDF link for granule {granule['umm']['GranuleUR']}")
    return links[0]


def clipped_path(product, date, clip_dir):
    """
    Path of the mosaicked, reprojected, clipped file for one date.
    """
    return os.path.join(clip_dir, f"{product}_{date:%Y%m%d}_clip.nc")


def reprojected_path(hdf_name, reproj_dir):
    """
    Path of the reprojected (unclipped) file for one granule.
    """
    return os.path.join(reproj_dir, os.path.splitext(os.path.basename(hdf_name))[0] + "_reproj.nc")


def open_granule(hdf_file, variables):
    """
    Open the requested layers of a MODIS HDF as a list of Datasets, one per resolution group
    (finest first), keeping the raw integer values and fill values.
    """
    groups = rxr.open_rasterio(hdf_file, variable=variables, mask_and_scale=False)
    groups = groups if isinstance(groups, list) else [groups]
    groups = [g.squeeze("band", drop=True) for g in groups]
    return sorted(groups, key=lambda g: abs(g.rio.resolution()[0]))


def reproject_granules(
        hdf_files, out_file, target_crs, variables, aoi=None, target_res=None,
        resampling="nearest", all_touched=False
        ):
    """
    Mosaic one or more same-date MODIS HDFs, reproject to target_crs, optionally clip to the AOI,
    and save all layers to a compressed NetCDF. Layers on coarser grids (e.g., 1 km state flags) are
    resampled with nearest neighbor onto the finest grid.
    """
    # Mosaic tiles in the native sinusoidal grid (tiles align exactly), per resolution group
    per_file = [open_granule(f, variables) for f in hdf_files]
    groups = [merge_datasets(list(g)) if len(g) > 1 else g[0] for g in zip(*per_file)]

    # Crop to the AOI bounds in the native projection first (much faster), then reproject and clip
    if aoi is not None:
        pad = 2 * max(abs(g.rio.resolution()[0]) for g in groups)
        xmin, ymin, xmax, ymax = aoi.to_crs(groups[0].rio.crs).total_bounds
        groups = [g.rio.clip_box(xmin - pad, ymin - pad, xmax + pad, ymax + pad) for g in groups]

    # Define the output grid from the finest layer, then match every layer to it
    fine = groups[0]
    res = target_res or abs(fine.rio.resolution()[0])
    first_var = list(fine.data_vars)[0]
    grid = fine[first_var].rio.reproject(target_crs, resolution=res)

    layers = {}
    for g in groups:
        for name, da in g.data_vars.items():
            if da.rio.nodata is None:
                fill = np.iinfo(da.dtype).max if np.issubdtype(da.dtype, np.integer) else np.nan
                da = da.rio.write_nodata(fill)
            method = Resampling[resampling] if name.startswith("sur_refl") else Resampling.nearest
            layers[name] = da.rio.reproject_match(grid, resampling=method)
    ds = layers[first_var].to_dataset(name=first_var).assign(layers)
    ds = ds.rio.write_crs(target_crs)

    if aoi is not None:
        ds = ds.rio.clip(aoi.to_crs(target_crs).geometry, all_touched=all_touched)

    # Add a time coordinate so outputs stack with xr.open_mfdataset
    _, date, _ = parse_granule_name(hdf_files[0])
    ds = ds.expand_dims(time=[date])
    ds.attrs = {
        "title": "MODIS/Terra Surface Reflectance Daily L2G Global (MOD09GA v061), reprojected",
        "source_files": ", ".join(os.path.basename(f) for f in hdf_files),
        "collection_concept_id": CONCEPT_ID,
        "note": "Raw integer values; apply each variable's scale_factor/add_offset attributes.",
    }

    # Keep raw integer packing; move fill values to encoding and compress
    encoding = {}
    for v in ds.data_vars:
        fill = ds[v].attrs.pop("_FillValue", ds[v].rio.nodata)
        ds[v].attrs.pop("grid_mapping", None)
        ds[v].encoding = {}
        encoding[v] = {"zlib": True, "complevel": 4, "_FillValue": fill, "dtype": ds[v].dtype}

    # Write the destination CRS last (spatial_ref variable + transform), and link every layer to it
    # with a grid_mapping attribute so GDAL/QGIS can read the CRS
    ds = ds.rio.write_crs(target_crs, grid_mapping_name="spatial_ref").rio.write_transform()
    for v in ds.data_vars:
        ds[v].attrs["grid_mapping"] = "spatial_ref"
    ds.load().to_netcdf(out_file, encoding=encoding)
    return out_file


def search_granules(aoi, start_date, end_date, months):
    """
    Search CMR for granules in the date range that fall on MODIS tiles intersecting the AOI.
    Returns {date: [granules]}.
    """
    tiles = tiles_from_aoi(aoi)
    print(f"MODIS tiles: {', '.join(tiles)}")
    w, s, e, n = aoi.to_crs(epsg=4326).total_bounds
    results = earthaccess.search_data(
        concept_id=CONCEPT_ID,
        temporal=(start_date, end_date),
        bounding_box=(float(w), float(s), float(e), float(n)),
        )

    by_date = defaultdict(list)
    for granule in results:
        _, date, tile = parse_granule_name(os.path.basename(hdf_url(granule)))
        if tile in tiles and date.month in months:
            by_date[date].append(granule)
    return dict(sorted(by_date.items()))


def download_modis_sr(
        out_dir, aoi_file, start_date, end_date, target_crs=None, months=range(1, 13),
        variables=VARIABLES, target_res=None, resampling="nearest", clip_to_aoi=True,
        all_touched=False, remove_unclipped=True, overwrite=False, dry_run=False, threads=4
        ):
    """
    Download MOD09GA granules covering aoi_file between start_date and end_date (inclusive),
    reproject to target_crs, and optionally clip to the AOI.
    """
    aoi = gpd.read_file(aoi_file)
    target_crs = target_crs or aoi.crs
    raw_dir = os.path.join(out_dir, "raw")
    proc_dir = os.path.join(out_dir, "clipped" if clip_to_aoi else "reprojected")
    for d in (raw_dir, proc_dir):
        os.makedirs(d, exist_ok=True)

    earthaccess.login()
    by_date = search_granules(aoi, start_date, end_date, months)
    n_granules = sum(len(g) for g in by_date.values())
    print(f"Found {n_granules} granules on {len(by_date)} dates")

    # Decide what still needs doing: skip dates whose processed output exists; reuse raw HDFs on disk
    jobs = []  # (date, [granules to download], [all raw paths], [output paths])
    for date, granules in by_date.items():
        names = [os.path.basename(hdf_url(g)) for g in granules]
        raw_paths = [os.path.join(raw_dir, n) for n in names]
        if clip_to_aoi:
            product, _, _ = parse_granule_name(names[0])
            outputs = [clipped_path(product, date, proc_dir)]
        else:
            outputs = [reprojected_path(n, proc_dir) for n in names]
        if not overwrite and all(os.path.exists(o) for o in outputs):
            continue
        to_download = [
            g for g, p in zip(granules, raw_paths) if overwrite or not os.path.exists(p)
            ]
        jobs.append((date, to_download, raw_paths, outputs))

    n_download = sum(len(j[1]) for j in jobs)
    print(
        f"{len(by_date) - len(jobs)} dates already processed; "
        f"{len(jobs)} dates to process ({n_download} granules to download)"
        )
    if dry_run:
        for date, to_download, _, _ in jobs:
            print(f"  {date:%Y-%m-%d}: {len(to_download)} to download")
        return

    failed = []
    for date, to_download, raw_paths, outputs in tqdm(jobs, desc="Downloading and reprojecting MOD09GA"):
        try:
            if to_download:
                earthaccess.download(to_download, local_path=raw_dir, threads=threads)
            missing = [p for p in raw_paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"Download failed: {', '.join(os.path.basename(m) for m in missing)}")

            if clip_to_aoi:
                reproject_granules(
                    raw_paths, outputs[0], target_crs, variables, aoi=aoi, target_res=target_res,
                    resampling=resampling, all_touched=all_touched
                    )
                if remove_unclipped:
                    for p in raw_paths:
                        os.remove(p)
            else:
                for raw, out in zip(raw_paths, outputs):
                    reproject_granules(
                        [raw], out, target_crs, variables, target_res=target_res,
                        resampling=resampling
                        )
        except Exception as err:
            # Don't leave a partial output that would be skipped on re-run
            for o in outputs:
                if os.path.exists(o):
                    os.remove(o)
            failed.append(date)
            tqdm.write(f"{date:%Y-%m-%d} failed: {err}")

    print(f"Outputs saved to {proc_dir}")
    if failed:
        print(f"{len(failed)} dates failed (re-run to retry): {', '.join(f'{d:%Y-%m-%d}' for d in failed)}")


def main():
    download_modis_sr(
        OUT_DIR, AOI_FILE, DATE_START, DATE_END, target_crs=TARGET_CRS, months=MONTHS,
        variables=VARIABLES, target_res=TARGET_RES, resampling=RESAMPLING, clip_to_aoi=CLIP_TO_AOI,
        all_touched=ALL_TOUCHED, remove_unclipped=REMOVE_UNCLIPPED, overwrite=OVERWRITE,
        dry_run=DRY_RUN, threads=THREADS
        )


if __name__ == "__main__":
    main()

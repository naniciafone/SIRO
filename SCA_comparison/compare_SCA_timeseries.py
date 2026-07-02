#!/usr/bin/env python
"""
Compare model- and satellite-derived snow-covered area (SCA) time series

Rainey Aberle (rainey.aberle@usace.army.mil)
Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL
July 2026
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from glob import glob
from tqdm import tqdm
import xrspatial


# ----- CONFIG -----
# I/O
base_dir = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
model_sca_dir = os.path.join(base_dir, "model_SCA")
model_names = ["HMS-EB", "HMS-TI", "iSnobal", "SnowModel"]
pss_sca_files = sorted(glob(os.path.join(base_dir, "PSS_SCA", "*.tif")))
dem_file = os.path.join(base_dir, "..", "DEMs", "USGS_2026_DEM_merged.tif")
aoi_file = os.path.join(base_dir, "..", "MCS_outline.gpkg")
out_dir = os.path.join(base_dir, "SCA_comparison_results")
os.makedirs(out_dir, exist_ok=True)

# Target resolution for all comparisons (meters)
TARGET_RES = 100

# PlanetScope-specific params
FSCA_THRESHOLD = 0.5                    # fSCA threshold after resampling
TREE_MAX = 0.5                          # Max tree percentage to consider "valid" after resampling
VALID_MIN = 0.5                         # Minimum percentage of classified pixels to consider "valid"
RUN_THRESHOLD_SWEEP = False             # Run confusion matrix threshold sensitivity loop
FSCA_SWEEP = np.arange(0.1, 1.01, 0.1)  # fSCA values to test
CHUNKS = {'x': 2048, 'y': 2048}         # Read PSS files in chunks to prevent RAM overload
CRS = "EPSG:32611"                      # Known CRS (projected, meters) for ALL rasters/grids
TIME_TOL = np.timedelta64(1, "D")       # Max time difference between PlanetScope and models for comparison
# classified image values
NODATA_VAL = 255
TREE_VAL = 2
SNOW_VAL = 1
NOSNOW_VAL = 0

# Iterations
TASKS = [1, 2]
SWE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

# Output files
PSS_FSCA_REGRID_FILE = os.path.join(out_dir, "fSCA_100m_PlanetScope.nc")
PSS_TOTALS_NATIVE_FILE = os.path.join(out_dir, "SCA_totals_native_PlanetScope.csv")
PSS_TOTALS_REGRID_FILE = os.path.join(out_dir, "SCA_totals_100m_PlanetScope.nc")
SCA_MODELED_FILE = os.path.join(out_dir, "compiled_SCA_totals_100m_modeled.nc")
CM_FILE = os.path.join(out_dir, "compiled_confusion_matrices.nc")
RECALL_FILE = os.path.join(out_dir, "compiled_recall_binned_elev_aspect.nc")

# ----- HELPER FUNCTIONS -----
def reproject_fractional_sca(da_native, target_grid):
    """
    Resample a native-resolution PlanetScope land cover mask to the target grid.

    Parameters
    ----------
    da_native : xr.DataArray
        Native raster with values: 0=no snow, 1=snow, 2=tree, 255=nodata
    target_grid : xr.DataArray
        The target grid to match

    Returns
    -------
    dict of xr.DataArray on the target grid:
        'fsca'        : snow / (snow + no_snow). NaN where no certain (snow/no-snow) area exists.
        'snow_frac'   : fraction of cell that is snow
        'tree_frac'   : fraction of cell that is tree
        'certain_frac': fraction of cell that is snow or no-snow
        'valid_frac'  : fraction of cell that is any classified pixel
    """
    # Mark nodata as NaN
    da = xr.where(da_native == NODATA_VAL, np.nan, da_native).astype("float32")

    def _indicator(value):
        ind = xr.where(np.isnan(da), np.nan, xr.where(da == value, 1.0, 0.0))
        ind = ind.rio.write_crs(CRS)
        ind = ind.rio.write_nodata(np.nan)
        return ind

    is_snow = _indicator(SNOW_VAL)
    is_nosnow = _indicator(NOSNOW_VAL)
    is_tree = _indicator(TREE_VAL)
    is_valid = xr.where(np.isnan(da), np.nan, 1.0).rio.write_crs(CRS)
    is_valid = is_valid.rio.write_nodata(np.nan)

    # Calculate fractional coverage of each class on the target grid
    snow_frac = is_snow.rio.reproject_match(target_grid, resampling='average')
    nosnow_frac = is_nosnow.rio.reproject_match(target_grid, resampling='average')
    tree_frac = is_tree.rio.reproject_match(target_grid, resampling='average')
    valid_frac = is_valid.rio.reproject_match(target_grid, resampling='average')

    # Calculate fSCA over known area only, i.e. tree-covered areas excluded.
    certain_frac = snow_frac + nosnow_frac
    fsca = xr.where(certain_frac > 0, snow_frac / certain_frac, np.nan)

    return {
        "fsca": fsca,
        "snow_frac": snow_frac,
        "tree_frac": tree_frac,
        "certain_frac": certain_frac,
        "valid_frac": valid_frac,
    }


def build_comparison_mask(
        tree_frac, certain_frac, valid_frac, tree_max=TREE_MAX, valid_min=VALID_MIN
        ):
    """
    Boolean mask on the target grid of cells valid for model/PSS comparison.
    """
    enough_valid = valid_frac >= valid_min
    not_too_treed = tree_frac <= tree_max
    has_certain = certain_frac > 0
    return enough_valid & not_too_treed & has_certain


def calculate_model_sca(model_binary, comp_mask, pixel_area):
    """
    Model area = binary model SCA (1=snow) summed over the common mask only.
    """
    m = model_binary.where(comp_mask)
    snow = xr.where(m == 1, 1.0, 0.0).where(comp_mask).sum(dim=["x", "y"]).data
    valid = xr.where(~np.isnan(m), 1.0, 0.0).sum(dim=["x", "y"]).data
    return float(snow * pixel_area), float(valid * pixel_area)


def calculate_planetscope_sca(snow_frac, certain_frac, valid_frac, comp_mask, pixel_area):
    """
    Compute PlanetScope SCA total and its uncertainty bounds for one timestamp
    within the comparison mask.

    Best estimate:
        SCA = sum(snow_frac * pixel_area) over masked cells.

    Uncertain area = the fraction of each masked cell whose snow status is NOT
    certain (i.e. trees or partially/edge-classified area) = (1 - certain_frac).
    Bounds assume the uncertain area is entirely no-snow (lower) or entirely
    snow (upper):
        SCA_lower = sum(snow_frac * pixel_area)
        SCA_upper = sum((snow_frac + uncertain_frac) * pixel_area)

    Returns dict of floats (m2).
    """
    sf = snow_frac.where(comp_mask)
    cf = certain_frac.where(comp_mask)
    vf = valid_frac.where(comp_mask)

    uncertain_frac = (1.0 - cf).clip(min=0.0)  # trees + unclassified/edge

    snow_area = float((sf * pixel_area).sum(dim=["x", "y"]).data)
    certain_area = float((cf * pixel_area).sum(dim=["x", "y"]).data)
    valid_area = float((vf * pixel_area).sum(dim=["x", "y"]).data)
    uncertain_area = float((uncertain_frac * pixel_area).sum(dim=["x", "y"]).data)

    return {
        "SCA_m2": snow_area,
        "SCA_lower_m2": snow_area,                      # uncertain -> all no-snow
        "SCA_upper_m2": snow_area + uncertain_area,     # uncertain -> all snow
        "uncertain_area_m2": uncertain_area,
        "certain_area_m2": certain_area,
        "valid_area_m2": valid_area,
    }


def load_and_reproject_pss_sca(
    pss_files: list = None,
    target_grid: xr.DataArray = None,
    out_file: str = None,
    chunks: dict = None,
) -> xr.Dataset:

    if not pss_files:
        raise FileNotFoundError("pss_files list is empty.")
    if chunks is None:
        chunks = CHUNKS

    if os.path.exists(out_file):
        print(f"Output file already exists, loading: {out_file}")
        return xr.open_dataset(out_file)

    fsca_list, snowf_list, treef_list, certf_list, validf_list = [], [], [], [], []
    for filepath in tqdm(sorted(pss_files)):
        with rxr.open_rasterio(filepath, chunks=chunks).squeeze() as da:
            da = da.rio.write_crs(CRS)

            out = reproject_fractional_sca(da, target_grid)

            date = os.path.basename(filepath).split("_")[0]
            dt = np.datetime64(date)
            for key, lst in (
                ("fsca", fsca_list), ("snow_frac", snowf_list),
                ("tree_frac", treef_list), ("certain_frac", certf_list),
                ("valid_frac", validf_list),
            ):
                arr = out[key].assign_coords(time=dt).expand_dims("time")
                lst.append(arr)

    def _concat(lst, name):
        c = xr.concat(lst, dim="time").sortby("time").rio.write_crs(CRS)
        c.name = name
        return c

    ds = xr.Dataset({
        "fsca": _concat(fsca_list, "fsca"),
        "snow_frac": _concat(snowf_list, "snow_frac"),
        "tree_frac": _concat(treef_list, "tree_frac"),
        "certain_frac": _concat(certf_list, "certain_frac"),
        "valid_frac": _concat(validf_list, "valid_frac"),
    })
    ds = ds.rio.write_crs(CRS)
    ds.attrs.update({
        "long_name": "Fractional snow-covered area fields on target grid",
        "fsca_definition": "snow / (snow + no_snow); trees EXCLUDED from denominator",
        "grid_resolution_m": TARGET_RES,
        "note": "Trees are not assumed snow-covered; they are excluded from comparison.",
    })

    ds.to_netcdf(out_file)
    print("Saved regridded PlanetScope fractional fields to:", out_file)
    return ds


def calculate_confusion_matrix(model_binary, ref_binary, comp_mask=None):
    """
    NOTE: relies on model_binary/ref_binary preserving NaN for nodata/invalid
    cells (do NOT pre-binarize NaN model cells to 0 before calling this).
    """
    m = model_binary.data.ravel().astype("float32")
    r = ref_binary.data.ravel().astype("float32")
    valid = ~np.isnan(m) & ~np.isnan(r)
    if comp_mask is not None:
        valid &= comp_mask.data.ravel().astype(bool)
    m = m[valid].astype(bool)
    r = r[valid].astype(bool)

    # Calculate True/False Positives/Negatives
    tp = int(np.sum(m & r))
    tn = int(np.sum(~m & ~r))
    fp = int(np.sum(m & ~r))
    fn = int(np.sum(~m & r))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def get_elev_bins(dem_da, bin_width=50):
    min_elev = np.nanmin(dem_da.data)
    max_elev = np.nanmax(dem_da.data)
    start = bin_width * (min_elev // bin_width)
    end = bin_width * (1 + (max_elev // bin_width))
    return np.arange(start, end + bin_width, bin_width)


def calculate_recall_with_terrain(
        pss_binary, model_binary, elev_da, aspect_da, elev_bins, aspect_bins, comp_mask
        ):
    elev_bin_idx = np.digitize(elev_da.values, elev_bins) - 1
    aspect_bin_idx = np.digitize(aspect_da.values, aspect_bins) - 1
    n_elev_bins = len(elev_bins) - 1
    n_aspect_bins = len(aspect_bins) - 1
    n_time = pss_binary.sizes["time"]

    recall = np.full((n_time, n_elev_bins, n_aspect_bins), np.nan)
    cmask = comp_mask.values if comp_mask is not None else np.ones_like(elev_da.values, bool)
    for i in range(n_time):
        sat = pss_binary.isel(time=i).values
        mod = model_binary.isel(time=i).values
        cell_mask_i = cmask[i] if cmask.ndim == 3 else cmask
        for j in range(n_elev_bins):
            for k in range(n_aspect_bins):
                mask = (elev_bin_idx == j) & (aspect_bin_idx == k) & cell_mask_i
                tp = np.nansum((sat[mask] == 1) & (mod[mask] == 1))
                fn = np.nansum((sat[mask] == 1) & (mod[mask] == 0))
                denom = tp + fn
                recall[i, j, k] = tp / denom if denom > 0 else np.nan

    recall_da = xr.DataArray(
        recall,
        dims=["time", "elev_bin", "aspect_bin"],
        coords={
            "time": pss_binary["time"],
            "elev_bin": np.arange(n_elev_bins),
            "aspect_bin": np.arange(n_aspect_bins),
        },
        name="recall",
    )
    recall_ds = xr.Dataset({"recall": recall_da})
    recall_ds["recall"].attrs["elev_bins"] = elev_bins
    recall_ds["recall"].attrs["aspect_bins"] = aspect_bins
    return recall_ds


def build_or_load_model_grid(model, target_grid_da):
    """
    Build (or load from cache) the regridded model SCA stack for ONE model.

    Returns an xr.DataArray "SCA" with dims (task, SWE_threshold_m, time, y, x),
    stored as uint16 with NODATA_VAL as fill. One NetCDF file per model:
    SCA_100m_gridded_{model}.nc
    """
    out_file = os.path.join(out_dir, f"SCA_100m_gridded_{model}.nc")

    if os.path.exists(out_file):
        print(f"Regridded model grid exists, loading: {os.path.basename(out_file)}")
        return xr.open_dataset(out_file)["SCA"]

    print(f"Reprojecting all Task/SWE layers for model: {model}")
    task_layers = []
    for task in TASKS:
        swe_layers = []
        for swe_thresh in SWE_THRESHOLDS:
            model_files = sorted(glob(os.path.join(
                model_sca_dir, f"{model}*Task{task}*SWEthresh{swe_thresh}m.nc")))
            if not model_files:
                print(f"  No files for {model}, Task {task}, SWE {swe_thresh} m")
                swe_layers.append(None)
                continue

            mf = model_files[0]
            with xr.open_dataset(mf) as model_ds_native:
                model_sca_native = model_ds_native["SCA"].squeeze()
                model_sca_native = xr.where(
                    model_sca_native == NODATA_VAL, np.nan, model_sca_native)
                model_sca_native = model_sca_native.rio.write_crs(CRS)
                # Nearest neighbor for binary/categorical data
                regrid = model_sca_native.rio.reproject_match(
                    target_grid_da, resampling='nearest'
                )
            regrid = regrid.assign_coords(SWE_threshold_m=swe_thresh)
            swe_layers.append(regrid)

        # Reconcile any missing SWE layers by matching the shape of a present one
        template = next((s for s in swe_layers if s is not None), None)
        if template is None:
            raise FileNotFoundError(f"No model files found at all for {model}, Task {task}")
        swe_layers = [
            s if s is not None
            else xr.full_like(template, np.nan).assign_coords(SWE_threshold_m=swe)
            for s, swe in zip(swe_layers, SWE_THRESHOLDS)
        ]
        task_stack = xr.concat(swe_layers, dim="SWE_threshold_m")
        task_stack = task_stack.assign_coords(task=task)
        task_layers.append(task_stack)

    # Stack over task -> dims (task, SWE_threshold_m, time, y, x)
    model_stack = xr.concat(task_layers, dim="task")

    # Store as uint16 with NODATA fill
    model_stack = xr.where(np.isnan(model_stack), NODATA_VAL, model_stack).astype(np.uint16)
    model_stack.name = "SCA"
    model_stack = model_stack.rio.write_crs(CRS)
    model_stack = model_stack.rio.write_nodata(NODATA_VAL)
    model_stack.attrs.update({
        "long_name": f"Regridded model SCA grids ({model})",
        "grid_resolution_m": TARGET_RES,
        "nodata": NODATA_VAL,
        "values": "0=no snow, 1=snow, 255=nodata",
    })

    model_stack.to_dataset(name="SCA").to_netcdf(
        out_file, encoding={"SCA": {"dtype": "uint16"}}
    )
    print(f"Saved regridded model grid: {os.path.basename(out_file)}")
    return model_stack


# ----- MAIN WORKFLOW -----
def main():
    os.makedirs(out_dir, exist_ok=True)

    # --- Calculate PlanetScope areas time series at native grid ---
    if os.path.exists(PSS_TOTALS_NATIVE_FILE):
        print(f"PlanetScope native-grid areas exist, loading: {PSS_TOTALS_NATIVE_FILE}")
        sca_pss_native_df = pd.read_csv(PSS_TOTALS_NATIVE_FILE)
    else:
        print("Calculating PlanetScope areas time series at native grid...")
        sca_pss_native_list = []
        for pss_sca_file in tqdm(pss_sca_files):
            with rxr.open_rasterio(pss_sca_file, masked=False, chunks=CHUNKS).squeeze() as da:
                da = da.rio.write_crs(CRS)
                res = da.rio.resolution()
                pixel_area_native = abs(res[0] * res[1])
                snow_area = float(xr.where(da == SNOW_VAL, 1, 0).sum().data) * pixel_area_native
                nosnow_area = float(xr.where(da == NOSNOW_VAL, 1, 0).sum().data) * pixel_area_native
                tree_area = float(xr.where(da == TREE_VAL, 1, 0).sum().data) * pixel_area_native
                nodata_area = float(xr.where(da == NODATA_VAL, 1, 0).sum().data) * pixel_area_native
            df = pd.DataFrame({
                "datetime": [os.path.basename(pss_sca_file).split("_")[0]],
                "SCA_PSS-native-grid_m2": [snow_area],
                "nosnow_area_PSS-native-grid_m2": [nosnow_area],
                "tree_area_PSS-native-grid_m2": [tree_area],
                "nodata_area_PSS-native-grid_m2": [nodata_area],
            }, index=[0])
            sca_pss_native_list += [df]
        sca_pss_native_df = pd.concat(sca_pss_native_list, ignore_index=True)
        sca_pss_native_df.to_csv(PSS_TOTALS_NATIVE_FILE, index=False, header=True)
        print(f"Saved to: {PSS_TOTALS_NATIVE_FILE}")

    # --- Create a new 100 m grid ---
    print("Creating new 100 m grid for comparisons")
    target_grid_file = os.path.join(out_dir, "target_grid.tif")
    if not os.path.exists(target_grid_file):
        # Load AOI
        aoi = gpd.read_file(aoi_file)
        aoi = aoi.to_crs(CRS)

        # Create "nice" grid with round numbers
        xmin, ymin, xmax, ymax = aoi.geometry.bounds.values[0]
        xmin_round = np.floor(xmin / TARGET_RES) * TARGET_RES
        xmax_round = np.ceil(xmax / TARGET_RES) * TARGET_RES
        ymin_round = np.floor(ymin / TARGET_RES) * TARGET_RES
        ymax_round = np.ceil(ymax / TARGET_RES) * TARGET_RES
        target_x = np.arange(xmin_round, xmax_round + TARGET_RES, TARGET_RES)
        target_y = np.arange(ymax_round, ymin_round - TARGET_RES, -TARGET_RES)  # Top-down for Y

        # Create dummy data array for reprojecting rasters
        target_grid_da = xr.DataArray(
            data=np.ones((len(target_y), len(target_x))),
            dims=["y", "x"],
            coords={"y": target_y, "x": target_x},
        ).rio.write_crs(CRS)
        target_grid_da = target_grid_da.rio.clip(
            geometries=aoi.geometry,
            crs=CRS,
            drop=False,
            all_touched=True
        )
        pixel_area = TARGET_RES ** 2

        # Save to file
        target_grid_da.rio.to_raster(target_grid_file)
        print("Target grid saved to:", target_grid_file)
    else:
        print("Target grid file already exists, loading.")
        target_grid_da = rxr.open_rasterio(target_grid_file, masked=True).squeeze()
        pixel_area = TARGET_RES ** 2

    # --- Calculate PlanetScope fractional areas at target grid ---
    if not os.path.exists(PSS_FSCA_REGRID_FILE):
        print("Regridding PlanetScope fSCA to target grid")
        pss_fsca_regrid = load_and_reproject_pss_sca(
            pss_sca_files, target_grid_da, PSS_FSCA_REGRID_FILE, chunks=CHUNKS
        )
    else:
        print("Regridded PlanetScope FSCA exist, skipping.")
        pss_fsca_regrid = xr.open_dataset(PSS_FSCA_REGRID_FILE)

    # --- Comparison mask ---
    comp_mask_ts = build_comparison_mask(
        pss_fsca_regrid["tree_frac"], pss_fsca_regrid["certain_frac"], pss_fsca_regrid["valid_frac"]
    )

    time_coord = pss_fsca_regrid["time"]
    n_time = pss_fsca_regrid.sizes["time"]

    # -----------------------------------------------------------------
    # PlanetScope SCA totals + uncertainty
    # -----------------------------------------------------------------
    if os.path.exists(PSS_TOTALS_REGRID_FILE):
        print(f"PlanetScope SCA totals exist, loading: {PSS_TOTALS_REGRID_FILE}")
    else:
        print("Calculating PlanetScope SCA totals and uncertain-area totals (target grid)...")
        keys = [
            "SCA_m2", "SCA_lower_m2", "SCA_upper_m2", 
            "uncertain_area_m2", "certain_area_m2", "valid_area_m2"
            ]
        pss_totals = {k: np.full(n_time, np.nan, dtype="float64") for k in keys}
        for i in tqdm(range(n_time)):
            comp_mask_i = comp_mask_ts.isel(time=i)
            res = calculate_planetscope_sca(
                pss_fsca_regrid["snow_frac"].isel(time=i),
                pss_fsca_regrid["certain_frac"].isel(time=i),
                pss_fsca_regrid["valid_frac"].isel(time=i),
                comp_mask_i,
                pixel_area,
            )
            for k in keys:
                pss_totals[k][i] = res[k]

        pss_totals_ds = xr.Dataset(
            {k: ("time", pss_totals[k]) for k in keys},
            coords={"time": time_coord.values},
        )
        pss_totals_ds["SCA_m2"].attrs.update({
            "long_name": "PlanetScope snow-covered area within comparison mask",
            "units": "m2",
        })
        pss_totals_ds["SCA_lower_m2"].attrs.update({
            "long_name": "Lower bound: uncertain (tree/unclassified) area treated as no-snow",
            "units": "m2",
        })
        pss_totals_ds["SCA_upper_m2"].attrs.update({
            "long_name": "Upper bound: uncertain (tree/unclassified) area treated as snow",
            "units": "m2",
        })
        pss_totals_ds["uncertain_area_m2"].attrs.update({
            "long_name": "Area of uncertain snow status (trees + unclassified/edge) within mask",
            "units": "m2",
        })
        pss_totals_ds.attrs.update({
            "long_name": "PlanetScope SCA totals with uncertainty bounds",
            "grid_resolution_m": TARGET_RES,
            "uncertainty_definition": (
                "Bounds bracket the best estimate by assigning the uncertain "
                "area (1 - certain_frac) entirely to no-snow (lower) or snow (upper)."
            ),
        })
        pss_totals_ds.to_netcdf(PSS_TOTALS_REGRID_FILE)
        print("Saved:", PSS_TOTALS_REGRID_FILE)

    # --- Elevation and aspect bins ---
    print("Calculating elevation and aspect bins from native DEM")
    with rxr.open_rasterio(dem_file, masked=True).squeeze() as dem_da:
        dem_regrid = dem_da.rio.reproject_match(target_grid_da, resampling='bilinear')
    aspect_regrid = xrspatial.aspect(dem_regrid)
    elev_bins = get_elev_bins(dem_regrid, bin_width=50)
    aspect_bins = np.arange(0, 361, 45)

    # -----------------------------------------------------------------
    # Accumulators for the consolidated outputs
    # -----------------------------------------------------------------
    sca_modeled = {} 
    cm_accum = {"TP": {}, "TN": {}, "FP": {}, "FN": {}}
    recall_accum = {}
    recall_elev_bins = elev_bins
    recall_aspect_bins = aspect_bins

    # Iterate MODEL -> TASK -> SWE so each model's grid file is built once
    for model in model_names:
        print(f"\n{'='*60}\nProcessing model: {model}\n{'='*60}")

        # Build or load the single per-model regridded dataset
        model_grid = build_or_load_model_grid(model, target_grid_da)  # (task, SWE, time, y, x)

        for task in TASKS:
            print(f"\n  Task {task}")
            for swe_thresh in tqdm(SWE_THRESHOLDS):
                try:
                    layer = model_grid.sel(task=task, SWE_threshold_m=swe_thresh)
                except KeyError:
                    print(f"  Missing layer for {model}, Task {task}, SWE {swe_thresh} m")
                    continue

                model_da = xr.where(layer == NODATA_VAL, np.nan, layer)
                model_da = model_da.rio.write_crs(CRS)

                # Match model timestamps to PlanetScope timestamps
                model_matched = model_da.reindex(
                    time=pss_fsca_regrid["time"], method="nearest", tolerance=TIME_TOL
                )

                key = (model, task, swe_thresh)

                # ----- SCA totals -----
                sca_arr = np.full(n_time, np.nan, dtype="float64")
                for i in range(n_time):
                    comp_mask_i = comp_mask_ts.isel(time=i)
                    m_i = model_matched.isel(time=i)
                    mod_snow, _mod_valid = calculate_model_sca(m_i, comp_mask_i, pixel_area)
                    sca_arr[i] = round(mod_snow)
                sca_modeled[key] = sca_arr

                # ----- Confusion matrix -----
                tp_arr = np.full(n_time, np.nan, dtype="float64")
                tn_arr = np.full(n_time, np.nan, dtype="float64")
                fp_arr = np.full(n_time, np.nan, dtype="float64")
                fn_arr = np.full(n_time, np.nan, dtype="float64")
                for i in range(n_time):
                    comp_mask_i = comp_mask_ts.isel(time=i)
                    fsca_i = pss_fsca_regrid["fsca"].isel(time=i)
                    model_i = model_matched.isel(time=i)

                    mod_bin = xr.where(np.isnan(model_i), np.nan, xr.where(model_i == 1, 1.0, 0.0))
                    pss_bin = xr.where(np.isnan(fsca_i), np.nan,
                                       xr.where(fsca_i >= FSCA_THRESHOLD, 1.0, 0.0))
                    cm = calculate_confusion_matrix(mod_bin, pss_bin, comp_mask_i)
                    tp_arr[i] = cm["TP"]
                    tn_arr[i] = cm["TN"]
                    fp_arr[i] = cm["FP"]
                    fn_arr[i] = cm["FN"]
                cm_accum["TP"][key] = tp_arr
                cm_accum["TN"][key] = tn_arr
                cm_accum["FP"][key] = fp_arr
                cm_accum["FN"][key] = fn_arr

                # ----- Elevation/aspect binned recall -----
                pss_bin_all = xr.where(
                    np.isnan(pss_fsca_regrid["fsca"]), np.nan,
                    xr.where(pss_fsca_regrid["fsca"] >= FSCA_THRESHOLD, 1.0, 0.0)
                )
                mod_bin_all = xr.where(
                    np.isnan(model_matched), np.nan, xr.where(model_matched == 1, 1.0, 0.0)
                )
                recall_terrain_ds = calculate_recall_with_terrain(
                    pss_bin_all, mod_bin_all, dem_regrid, aspect_regrid,
                    elev_bins, aspect_bins, comp_mask_ts
                )
                recall_accum[key] = recall_terrain_ds["recall"].values

    # ----- Build coordinate arrays for the consolidated files -----
    models_coord = list(model_names)
    tasks_coord = list(TASKS)
    swe_coord = list(SWE_THRESHOLDS)
    n_model = len(models_coord)
    n_task = len(tasks_coord)
    n_swe = len(swe_coord)

    # ----- Save output files -----
    # Modeled SCA totals
    print("\nWriting compiled modeled SCA totals...")
    sca_data = np.full((n_model, n_task, n_swe, n_time), np.nan, dtype="float64")
    for (model, task, swe), arr in sca_modeled.items():
        mi = models_coord.index(model)
        ti = tasks_coord.index(task)
        si = swe_coord.index(swe)
        sca_data[mi, ti, si, :] = arr

    sca_ds = xr.Dataset(
        {"SCA": (["model", "task", "SWE_threshold_m", "time"], sca_data)},
        coords={
            "model": models_coord,
            "task": tasks_coord,
            "SWE_threshold_m": swe_coord,
            "time": time_coord.values,
        },
    )
    sca_ds["SCA"].attrs.update({
        "long_name": "Modeled snow-covered area within comparison mask",
        "units": "m2",
        "grid_resolution_m": TARGET_RES,
        "fsca_threshold": FSCA_THRESHOLD,
        "error_bar_note": "Model spread across SWE_threshold_m gives the model error bars.",
    })
    sca_ds.to_netcdf(SCA_MODELED_FILE)
    print("Saved:", SCA_MODELED_FILE)

    # Confusion matrices
    print("Writing compiled confusion matrices...")
    cm_vars = {}
    for name in ("TP", "TN", "FP", "FN"):
        data = np.full((n_model, n_task, n_swe, n_time), np.nan, dtype="float64")
        for (model, task, swe), arr in cm_accum[name].items():
            mi = models_coord.index(model)
            ti = tasks_coord.index(task)
            si = swe_coord.index(swe)
            data[mi, ti, si, :] = arr
        cm_vars[name] = (["model", "task", "SWE_threshold_m", "time"], data)

    cm_ds = xr.Dataset(
        cm_vars,
        coords={
            "model": models_coord,
            "task": tasks_coord,
            "SWE_threshold_m": swe_coord,
            "time": time_coord.values,
        },
    )
    cm_ds.attrs.update({
        "long_name": "Confusion matrix counts (model vs. PlanetScope fSCA)",
        "fsca_threshold": FSCA_THRESHOLD,
        "grid_resolution_m": TARGET_RES,
        "definition": "TP/TN/FP/FN of binary model SCA vs. binary PlanetScope fSCA within comparison mask",
    })
    cm_ds.to_netcdf(CM_FILE)
    print("Saved:", CM_FILE)

    # Recall with terrain
    print("Writing compiled recall with terrain...")
    n_elev_bins = len(recall_elev_bins) - 1
    n_aspect_bins = len(recall_aspect_bins) - 1
    recall_data = np.full(
        (n_model, n_task, n_swe, n_time, n_elev_bins, n_aspect_bins),
        np.nan, dtype="float64"
    )
    for (model, task, swe), arr in recall_accum.items():
        mi = models_coord.index(model)
        ti = tasks_coord.index(task)
        si = swe_coord.index(swe)
        recall_data[mi, ti, si, :, :, :] = arr

    recall_ds = xr.Dataset(
        {"recall": (
            ["model", "task", "SWE_threshold_m", "time", "elev_bin", "aspect_bin"],
            recall_data
        )},
        coords={
            "model": models_coord,
            "task": tasks_coord,
            "SWE_threshold_m": swe_coord,
            "time": time_coord.values,
            "elev_bin": np.arange(n_elev_bins),
            "aspect_bin": np.arange(n_aspect_bins),
        },
    )
    recall_ds["recall"].attrs["elev_bins"] = recall_elev_bins
    recall_ds["recall"].attrs["aspect_bins"] = recall_aspect_bins
    recall_ds["recall"].attrs["fsca_threshold"] = FSCA_THRESHOLD
    recall_ds["recall"].attrs["long_name"] = "Recall (model vs. PlanetScope) binned by elevation and aspect"
    recall_ds.to_netcdf(RECALL_FILE)
    print("Saved:", RECALL_FILE)

    print("\nDone! :3")


if __name__ == "__main__":
    main()
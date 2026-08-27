#!/usr/bin/env python
"""
Compare model- and satellite-derived snow-covered area (SCA) time series.
Updated to use SPIReS-MODIS/Terra fSCA product for validation.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL
July 2026
"""

import os
import numpy as np
import xarray as xr
import rioxarray as rxr
from glob import glob
from tqdm import tqdm
import xrspatial
import geopandas as gpd


# ----- CONFIG -----
# I/O
BASE_DIR = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
MODEL_SCA_DIR = os.path.join(BASE_DIR, "model_SCA")
FINE_RES_MODELS = ["iSnobal", "SnowModel"]  # 100 m native resolution
COARSE_RES_MODELS = ["HMS-EB", "HMS-TI"]    # 2 km native resolution
MODEL_NAMES = FINE_RES_MODELS + COARSE_RES_MODELS

DEM_FILE = os.path.join(BASE_DIR, "..", "DEMs", "USGS_2026_DEM_merged.tif")
AOI_FILE = os.path.join(BASE_DIR, "..", "MCS_outline.gpkg")
OUT_DIR = os.path.join(BASE_DIR, "SCA_comparison_results_SPIReS")
SPIRES_FSCA_FILE = os.path.join(OUT_DIR, "SPIReS_fSCA.nc")
os.makedirs(OUT_DIR, exist_ok=True)


# SPIReS product deets
CHUNKS = {'x': 1024, 'y': 1024}
CRS = "EPSG:32611"
TIME_TOL = np.timedelta64(12, "h")  # Max time difference for temporal sampling
NODATA_VAL = 255

# Iterations
TASKS = [1, 2]
SWE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

# Output files
METRICS_FILE = os.path.join(OUT_DIR, "compiled_performance_metrics.nc")
BINNED_METRICS_FILE = os.path.join(OUT_DIR, "compiled_metrics_binned_elev_aspect.nc")

# Melt-out date settings
MELT_OUT_FSCA_THRESH = 0.10   # fSCA (or binary SCA) threshold defining "snow covered"
WATER_YEAR_START_MONTH = 10   # water year begins Oct 1
MELT_OUT_METRICS_FILE = os.path.join(OUT_DIR, "compiled_melt_out_metrics.nc")
MELT_OUT_GRIDS_FILE = os.path.join(OUT_DIR, "melt_out_dates_gridded.nc")
MELT_OUT_BINNED_FILE = os.path.join(OUT_DIR, "compiled_melt_out_binned_elev_aspect.nc")
SAVE_MELT_OUT_GRIDS = True    # also save full per-pixel melt-out DOWY grids (larger file)


# ----- HELPER FUNCTIONS -----
def get_elev_bins(dem_da, bin_width=100):
    """Generate elevation bins for stratified analysis."""
    min_elev = np.nanmin(dem_da.data)
    max_elev = np.nanmax(dem_da.data)
    start = bin_width * (min_elev // bin_width)
    end = bin_width * (1 + (max_elev // bin_width))
    return np.arange(start, end + bin_width, bin_width)


def calculate_fsca_metrics(model_fsca, ref_fsca):
    """
    Calculate total fSCA performance metrics for fine-resolution models.
    Metrics are scaled to 0-100 where applicable.

    Parameters
    ----------
    model_fsca : xr.DataArray
        Model fractional snow cover (0-1)
    ref_fsca : xr.DataArray
        Reference (SPIReS) fractional snow cover (0-1)

    Returns
    -------
    xr.Dataset : Performance metrics as time series
    """

    # Calculate error metrics
    error = (model_fsca - ref_fsca) * 100
    rmse = ((error**2).mean(dim=['x', 'y']) ** 0.5)
    mae = (abs(error).mean(dim=['x', 'y']))
    bias = error.mean(dim=['x', 'y'])

    # Basin-averaged fSCA errors and correlation
    model_fsca_mean = model_fsca.mean(dim=['x', 'y']) 
    ref_fsca_mean = ref_fsca.mean(dim=['x', 'y'])
    fsca_average_error = (model_fsca_mean - ref_fsca_mean) * 100
    
    # Calculate correlation as a time series
    def _calculate_corr(a, b):
        # Flatten and remove NaNs
        a_flat = a.flatten()
        b_flat = b.flatten()
        valid_mask = ~np.isnan(a_flat) & ~np.isnan(b_flat)
        a_valid = a_flat[valid_mask]
        b_valid = b_flat[valid_mask]

        if a_valid.size < 2 or np.all(a_valid == a_valid[0]) or np.all(b_valid == b_valid[0]):
             return np.nan
        return np.corrcoef(a_valid, b_valid)[0, 1]

    correlation_list = []
    for t_idx in range(model_fsca.sizes['time']):
        model_slice = model_fsca.isel(time=t_idx).values
        ref_slice = ref_fsca.isel(time=t_idx).values
        correlation_list.append(_calculate_corr(model_slice, ref_slice))

    correlation = xr.DataArray(
        correlation_list, 
        dims=["time"], 
        coords={"time": model_fsca.time}
    )

    return xr.Dataset({
        'RMSE': rmse,
        'MAE': mae,
        'bias': bias,
        'fSCA_average_error': fsca_average_error,
        'correlation': correlation,
        })



def calculate_partial_credit_error(model_binary, ref_fsca):
    """
    Calculate partial credit error for coarse-resolution binary SCA,
    where 0 = perfect agreement and 1 = no agreement.

    For each model pixel:
    - If model = 1 (snow): error = 1 - mean(ref_fsca)
    - If model = 0 (no snow): error = mean(ref_fsca)

    Parameters
    ----------
    model_binary : xr.DataArray
        Binary model SCA (0 or 1), dims (time, y, x)
    ref_fsca : xr.DataArray
        Reference fSCA (0-1), dims (time, y, x)

    Returns
    -------
    xr.Dataset : Partial credit error as a time series
    """

    # Create masks for valid data
    valid_mask = ~np.isnan(model_binary) & ~np.isnan(ref_fsca)

    # Snow and no-snow masks
    snow_mask = (model_binary == 1) & valid_mask
    nosnow_mask = (model_binary == 0) & valid_mask

    # Calculate errors
    # For snow pixels: error = 1 - ref_fsca
    # For no-snow pixels: error = ref_fsca
    errors = xr.where(
        snow_mask, 1 - ref_fsca, xr.where(nosnow_mask, ref_fsca, np.nan)
        )

    # Overall partial credit error, scaled to 0-100
    partial_credit_error = errors.mean(dim=['x', 'y']) * 100

    return xr.Dataset({'partial_credit_error': partial_credit_error})


def calculate_metrics_binned(
        model_data, ref_fsca, elev_da, aspect_da, elev_bins, aspect_bins, is_binary=False
        ):
    """
    Calculate performance metrics binned by elevation and aspect.

    Parameters
    ----------
    model_data : xr.DataArray
        Model fSCA (fine-res) or binary SCA (coarse-res)
    ref_fsca : xr.DataArray
        Reference fSCA
    elev_da, aspect_da : xr.DataArray
        Elevation and aspect grids
    elev_bins, aspect_bins : array-like
        Bin edges
    is_binary : bool
        If True, use partial credit error; if False, use fSCA metrics

    Returns
    -------
    xr.Dataset with binned metrics
    """
    elev_bin_idx = np.digitize(elev_da.values, elev_bins) - 1
    aspect_bin_idx = np.digitize(aspect_da.values, aspect_bins) - 1
    n_elev_bins = len(elev_bins) - 1
    n_aspect_bins = len(aspect_bins) - 1
    n_time = model_data.sizes["time"]

    if is_binary:
        metric_names = ["partial_credit_error", "n_pixels"]
        metrics = {k: np.full((n_time, n_elev_bins, n_aspect_bins), np.nan)
                   for k in metric_names}
    else:
        metric_names = ["RMSE", "MAE", "correlation", "n_pixels"]
        metrics = {k: np.full((n_time, n_elev_bins, n_aspect_bins), np.nan)
                   for k in metric_names}

    for i in range(n_time):
        mod = model_data.isel(time=i).values
        ref = ref_fsca.isel(time=i).values

        for j in range(n_elev_bins):
            for k in range(n_aspect_bins):
                bin_mask = (elev_bin_idx == j) & (aspect_bin_idx == k)

                if not np.any(bin_mask):
                    continue

                mod_bin = mod[bin_mask]
                ref_bin = ref[bin_mask]
                valid = ~np.isnan(mod_bin) & ~np.isnan(ref_bin)

                if not np.any(valid):
                    continue

                mod_valid = mod_bin[valid]
                ref_valid = ref_bin[valid]

                if is_binary:
                    snow_mask = mod_valid == 1
                    errors = np.where(snow_mask, 1 - ref_valid, ref_valid)
                    metrics["partial_credit_error"][i, j, k] = np.mean(errors) * 100
                    metrics["n_pixels"][i, j, k] = len(mod_valid)
                else:
                    diff = mod_valid - ref_valid
                    metrics["RMSE"][i, j, k] = np.sqrt(np.mean(diff**2)) * 100
                    metrics["MAE"][i, j, k] = np.mean(np.abs(diff)) * 100
                    if np.std(mod_valid) > 0 and np.std(ref_valid) > 0:
                        metrics["correlation"][i, j, k] = np.corrcoef(mod_valid, ref_valid)[0, 1]
                    metrics["n_pixels"][i, j, k] = len(mod_valid)

    # Create dataset
    data_vars = {}
    for name in metric_names:
        data_vars[name] = (["time", "elev_bin", "aspect_bin"], metrics[name])

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": model_data["time"],
            "elev_bin": np.arange(n_elev_bins),
            "aspect_bin": np.arange(n_aspect_bins),
        }
    )
    ds.attrs["elev_bins"] = elev_bins.tolist()
    ds.attrs["aspect_bins"] = aspect_bins.tolist()

    return ds


def get_water_year(time_da, start_month=WATER_YEAR_START_MONTH):
    """
    Assign a water year label to each timestamp.
    Water year N runs from Oct 1 of year N-1 through Sep 30 of year N.

    Parameters
    ----------
    time_da : xr.DataArray of datetime64
    start_month : int
        Calendar month (1-12) that begins the water year.

    Returns
    -------
    xr.DataArray of int, same shape as time_da
    """
    return xr.where(
        time_da.dt.month >= start_month,
        time_da.dt.year + 1,
        time_da.dt.year,
    )


def calculate_melt_out_doy(fsca_da, thresh=MELT_OUT_FSCA_THRESH):
    """
    Per-pixel melt-out day-of-water-year (DOWY) for a single water year of fSCA/SCA data.

    Melt-out is defined as the LAST time step in the record at which a pixel is
    snow covered (fsca >= thresh). Taking the last (not first) qualifying
    observation means a spurious mid-winter dip below threshold (cloud gap,
    brief melt/refreeze) does not get mistaken for melt-out, as long as the
    pixel is observed snow-covered again later in the season.

    Parameters
    ----------
    fsca_da : xr.DataArray, dims (time, y, x)
        fSCA or binary SCA for a SINGLE water year (time should already be
        subset to one water year, e.g. via get_water_year + .sel/.where).
    thresh : float
        fSCA value at/above which a pixel counts as snow covered. Works for
        both continuous fSCA (fine-res models, SPIReS) and binary SCA
        (coarse-res models) as long as 0 < thresh < 1.

    Returns
    -------
    xr.DataArray, dims (y, x)
        Day-of-water-year (1 = start_month day 1) of the last snow-covered
        observation. NaN where a pixel is never observed snow-covered that
        water year, OR where it is still snow-covered at the final time step
        of the record (melt-out unresolved/censored -- make sure the input
        time series extends through the end of the ablation season).
    """
    is_snow = fsca_da >= thresh  # NaN treated as False (not a confirmed snow obs)

    dowy = xr.DataArray(
        np.arange(1, fsca_da.sizes["time"] + 1),
        dims="time",
        coords={"time": fsca_da["time"]},
    )

    melt_out_dowy = dowy.where(is_snow).max(dim="time", skipna=True)

    # Flag as unresolved (censored) if still snow-covered at the very last obs
    still_snow_at_end = is_snow.isel(time=-1)
    melt_out_dowy = melt_out_dowy.where(~still_snow_at_end)

    return melt_out_dowy


def calculate_melt_out_metrics(model_dowy, ref_dowy):
    """
    Pixel-wise melt-out date error metrics (in days) for one water year.

    Parameters
    ----------
    model_dowy, ref_dowy : xr.DataArray, dims (y, x)
        Melt-out day-of-water-year, e.g. from calculate_melt_out_doy. Should
        already be restricted to the same set of valid pixels (see the
        SPIReS-snow-covered masking applied in main()).

    Returns
    -------
    xr.Dataset : scalar bias/MAE/RMSE (days, model - ref) and correlation
    """
    error = model_dowy - ref_dowy
    valid = ~np.isnan(error)
    n_pixels = int(valid.sum().item())

    if n_pixels < 2:
        bias = mae = rmse = corr = np.nan
    else:
        err_valid = error.values[valid.values]
        bias = float(np.mean(err_valid))
        mae = float(np.mean(np.abs(err_valid)))
        rmse = float(np.sqrt(np.mean(err_valid**2)))
        mod_valid = model_dowy.values[valid.values]
        ref_valid = ref_dowy.values[valid.values]
        if np.std(mod_valid) > 0 and np.std(ref_valid) > 0:
            corr = float(np.corrcoef(mod_valid, ref_valid)[0, 1])
        else:
            corr = np.nan

    return xr.Dataset({
        "melt_out_bias_days": bias,
        "melt_out_MAE_days": mae,
        "melt_out_RMSE_days": rmse,
        "melt_out_correlation": corr,
        "melt_out_n_pixels": n_pixels,
    })


def calculate_melt_out_metrics_binned(model_dowy, ref_dowy, elev_da, aspect_da, elev_bins, aspect_bins):
    """
    Elevation/aspect-binned melt-out date error metrics (in days) for one water year.

    Unlike calculate_metrics_binned (fSCA), there's no time loop here -- melt-out
    date is a single value per pixel per water year, not a time series.

    Parameters
    ----------
    model_dowy, ref_dowy : xr.DataArray, dims (y, x)
        Melt-out day-of-water-year, already restricted to the same valid pixels
        (e.g. via the SPIReS-snow mask applied in main()).
    elev_da, aspect_da : xr.DataArray, dims (y, x)
        Elevation and aspect grids, same grid as model_dowy/ref_dowy.
    elev_bins, aspect_bins : array-like
        Bin edges.

    Returns
    -------
    xr.Dataset with dims (elev_bin, aspect_bin): bias/MAE/RMSE (days, model - ref)
    and pixel counts.
    """
    elev_bin_idx = np.digitize(elev_da.values, elev_bins) - 1
    aspect_bin_idx = np.digitize(aspect_da.values, aspect_bins) - 1
    n_elev_bins = len(elev_bins) - 1
    n_aspect_bins = len(aspect_bins) - 1

    metric_names = ["melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days", "melt_out_n_pixels"]
    metrics = {k: np.full((n_elev_bins, n_aspect_bins), np.nan) for k in metric_names}

    mod = model_dowy.values
    ref = ref_dowy.values

    for j in range(n_elev_bins):
        for k in range(n_aspect_bins):
            bin_mask = (elev_bin_idx == j) & (aspect_bin_idx == k)
            if not np.any(bin_mask):
                continue

            mod_bin = mod[bin_mask]
            ref_bin = ref[bin_mask]
            valid = ~np.isnan(mod_bin) & ~np.isnan(ref_bin)
            mod_valid = mod_bin[valid]
            ref_valid = ref_bin[valid]

            if mod_valid.size == 0:
                continue

            diff = mod_valid - ref_valid
            metrics["melt_out_bias_days"][j, k] = np.mean(diff)
            metrics["melt_out_MAE_days"][j, k] = np.mean(np.abs(diff))
            metrics["melt_out_RMSE_days"][j, k] = np.sqrt(np.mean(diff**2))
            metrics["melt_out_n_pixels"][j, k] = mod_valid.size

    data_vars = {name: (["elev_bin", "aspect_bin"], metrics[name]) for name in metric_names}
    ds = xr.Dataset(
        data_vars,
        coords={"elev_bin": np.arange(n_elev_bins), "aspect_bin": np.arange(n_aspect_bins)},
    )
    return ds


# ----- MAIN WORKFLOW -----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load and preprocess SPIReS fSCA ---
    print("Loading and preparing SPIReS fSCA...")
    spires_ds = xr.open_dataset(SPIRES_FSCA_FILE, mask_and_scale=False)
    spires_fsca_raw = spires_ds["snow_fraction"]
    spires_fsca = spires_fsca_raw.where(spires_fsca_raw != NODATA_VAL) / 100.0
    spires_fsca = spires_fsca.rio.write_crs(CRS)
    n_time = len(spires_fsca.time.data)
    aoi = gpd.read_file(AOI_FILE)

    # --- Load terrain data ---
    print("Loading and reprojecting DEM for terrain analysis...")
    with rxr.open_rasterio(DEM_FILE, masked=True).squeeze() as dem_da:
        dem_regrid = dem_da.rio.reproject_match(spires_fsca, resampling='bilinear')
        dem_regrid = dem_regrid.rio.clip(aoi.geometry)
    aspect_regrid = xrspatial.aspect(dem_regrid)
    elev_bins = get_elev_bins(dem_regrid, bin_width=100)
    aspect_bins = np.arange(0, 361, 45)

    # Metrics for all models
    all_metrics = {
        "RMSE": {}, "MAE": {}, "correlation": {}, "bias": {}, "fSCA_average_error": {}, "partial_credit_error": {}
    }
    binned_metrics = {}

    # --- Precompute SPIReS melt-out dates per water year (used to mask all models) ---
    print("Calculating SPIReS melt-out dates by water year...")
    water_year = get_water_year(spires_fsca["time"])
    spires_fsca = spires_fsca.assign_coords(water_year=("time", water_year.data))
    water_years = np.unique(water_year.data)

    spires_melt_out_by_wy = {}   # wy -> (y, x) DataArray of SPIReS melt-out DOWY
    spires_snow_mask_by_wy = {}  # wy -> (y, x) boolean, True where SPIReS ever saw snow that WY
    for wy in water_years:
        spires_wy = spires_fsca.sel(time=spires_fsca["water_year"] == wy)
        wy_doy = calculate_melt_out_doy(spires_wy, thresh=MELT_OUT_FSCA_THRESH)
        spires_melt_out_by_wy[wy] = wy_doy
        spires_snow_mask_by_wy[wy] = wy_doy.notnull()

    melt_out_metrics = {}  # (model, task, swe_thresh, wy) -> xr.Dataset of scalar metrics
    melt_out_grids = {}    # (model, task, swe_thresh, wy) -> (y, x) DataArray of model melt-out DOWY
    melt_out_binned = {}   # (model, task, swe_thresh, wy) -> xr.Dataset binned by elevation/aspect

    # --- Process each model ---
    print("Processing each model...")
    for model in MODEL_NAMES:
        model_matched_file = os.path.join(OUT_DIR, f"fSCA_regridded_{model}.nc")

        if os.path.exists(model_matched_file):
            print(f"Loading resampled fSCA for {model} from file...")
            combined_fsca = xr.open_dataset(model_matched_file)
        else:
            print(f"Creating resampled fSCA file for {model}...")
            all_model_fsca = []
            for task in TASKS:
                for swe_thresh in tqdm(SWE_THRESHOLDS, desc=f"{model} Task {task}"):
                    model_file = os.path.join(
                        MODEL_SCA_DIR,
                        f"{model}_SCA_timeseries_Task{task}_SWEthresh{swe_thresh}m.nc"
                    )
                    if not os.path.exists(model_file):
                        print(f"File not found, skipping: {model_file}")
                        continue
                    
                    with xr.open_dataset(model_file, mask_and_scale=True) as model_ds:
                        model_ds = xr.where(model_ds == 255, np.nan, model_ds)
                        model_matched = model_ds['SCA'].reindex(
                            time=spires_fsca['time'], method='nearest', tolerance=TIME_TOL
                        )
                        model_matched = xr.where(model_matched == NODATA_VAL, np.nan, model_matched).rio.write_crs(CRS)
                    
                    resample_method = 'average' if model in FINE_RES_MODELS else 'nearest'
                    model_fsca_da = model_matched.rio.reproject_match(
                        spires_fsca, resampling=resample_method
                    )
                    if model in FINE_RES_MODELS:
                        model_fsca_da = model_fsca_da.clip(0, 1)
                    
                    model_fsca_da = model_fsca_da.rio.clip(aoi.geometry)
                    model_fsca_da.name = "fSCA"
                    model_fsca_da = model_fsca_da.expand_dims(
                        {'task': [task], 'SWE_threshold_m': [swe_thresh]}
                    )
                    all_model_fsca.append(model_fsca_da)

            if not all_model_fsca:
                print(f"No data processed for model {model}, skipping.")
                continue

            combined_fsca = xr.combine_by_coords(all_model_fsca)
            combined_fsca.to_netcdf(model_matched_file)
            print("Resampled model fSCA saved to:", model_matched_file)

        # combined_fsca['time'] == spires_fsca['time'] (reindexed to it above), so the
        # water years computed from spires_fsca line up directly
        combined_fsca = combined_fsca.assign_coords(water_year=("time", water_year.data))

        # Now iterate through the combined data to calculate metrics
        for task in combined_fsca.task.values:
            for swe_thresh in combined_fsca.SWE_threshold_m.values:
                model_fsca = combined_fsca['fSCA'].sel(task=task, SWE_threshold_m=swe_thresh)
                key = (model, task, swe_thresh)

                if model in FINE_RES_MODELS:
                    metrics_ds = calculate_fsca_metrics(model_fsca, spires_fsca)
                    for metric_name in metrics_ds.data_vars:
                         if metric_name in all_metrics:
                            all_metrics[metric_name][key] = metrics_ds[metric_name].values
                    binned_ds = calculate_metrics_binned(
                        model_fsca, spires_fsca, dem_regrid, aspect_regrid,
                        elev_bins, aspect_bins, is_binary=False
                    )
                    binned_metrics[key] = binned_ds
                else:
                    scores_ds = calculate_partial_credit_error(model_fsca, spires_fsca)
                    all_metrics["partial_credit_error"][key] = scores_ds['partial_credit_error'].values
                    binned_ds = calculate_metrics_binned(
                        model_fsca, spires_fsca, dem_regrid, aspect_regrid,
                        elev_bins, aspect_bins, is_binary=True
                    )
                    binned_metrics[key] = binned_ds

        # --- Melt-out date comparison, masked to pixels SPIReS observed as snow-covered ---
        for wy in water_years:
            spires_snow_mask = spires_snow_mask_by_wy[wy]
            ref_dowy = spires_melt_out_by_wy[wy].where(spires_snow_mask)
            combined_fsca_wy = combined_fsca.sel(time=combined_fsca["water_year"] == wy)

            if combined_fsca_wy.sizes["time"] == 0:
                continue

            for task in combined_fsca_wy.task.values:
                for swe_thresh in combined_fsca_wy.SWE_threshold_m.values:
                    model_fsca_wy = combined_fsca_wy["fSCA"].sel(task=task, SWE_threshold_m=swe_thresh)

                    # Restrict to pixels SPIReS observed as snow-covered this water year
                    # (this is what excludes canopy-obscured / never-snow-per-SPIReS pixels)
                    model_fsca_masked = model_fsca_wy.where(spires_snow_mask)
                    model_dowy = calculate_melt_out_doy(model_fsca_masked, thresh=MELT_OUT_FSCA_THRESH)

                    key = (model, task, swe_thresh, wy)
                    melt_out_metrics[key] = calculate_melt_out_metrics(model_dowy, ref_dowy)
                    melt_out_binned[key] = calculate_melt_out_metrics_binned(
                        model_dowy, ref_dowy, dem_regrid, aspect_regrid, elev_bins, aspect_bins
                    )
                    if SAVE_MELT_OUT_GRIDS:
                        melt_out_grids[key] = model_dowy

    # --- Save outputs ---
    print("\nSaving performance metrics...")

    # Compiled performance metrics
    metric_names = list(all_metrics.keys())
    metrics_data = {}
    for metric in metric_names:
        # Check if the metric has any calculated values
        if not all_metrics[metric]:
            continue
        data = np.full((len(MODEL_NAMES), len(TASKS), len(SWE_THRESHOLDS), n_time), np.nan)
        for (model, task, swe), arr in all_metrics[metric].items():
            mi = MODEL_NAMES.index(model)
            ti = TASKS.index(task)
            si = SWE_THRESHOLDS.index(swe)
            # Ensure the array slice is compatible
            if arr.ndim == 1 and arr.shape[0] == n_time:
                data[mi, ti, si, :] = arr
        metrics_data[metric] = (["model", "task", "SWE_threshold_m", "time"], data)

    metrics_ds = xr.Dataset(
        metrics_data,
        coords={
            "model": MODEL_NAMES,
            "task": TASKS,
            "SWE_threshold_m": SWE_THRESHOLDS,
            "time": spires_fsca.time.data,
        }
    )
    metrics_ds.attrs.update({
        "description": "Performance metrics for all models vs. SPIReS fSCA.",
        "note_fine": "Fine-res models (RMSE, MAE, correlation are valid). Errors are on a 0-100 scale.",
        "note_coarse": "Coarse-res models (partial_credit_error is valid). Error is on a 0-100 scale.",
    })
    metrics_ds.to_netcdf(METRICS_FILE)
    print("Compiled performance metrics saved to:", METRICS_FILE)

    # Terrain-binned metrics
    n_elev = len(elev_bins) - 1
    n_aspect = len(aspect_bins) - 1
    binned_data = {}
    all_binned_metric_names = {"RMSE", "MAE", "correlation", "partial_credit_error", "n_pixels"}

    for metric in all_binned_metric_names:
        # Check if any binned data exists for this metric
        has_metric = any(metric in ds for ds in binned_metrics.values())
        if not has_metric:
            continue
            
        data = np.full((len(MODEL_NAMES), len(TASKS), len(SWE_THRESHOLDS), n_time, n_elev, n_aspect), np.nan)
        for (model, task, swe), ds in binned_metrics.items():
            mi = MODEL_NAMES.index(model)
            ti = TASKS.index(task)
            si = SWE_THRESHOLDS.index(swe)
            if metric in ds:
                data[mi, ti, si, :, :, :] = ds[metric].values
        binned_data[metric] = (
            ["model", "task", "SWE_threshold_m", "time", "elev_bin", "aspect_bin"], data
        )

    binned_ds = xr.Dataset(
        binned_data,
        coords={
            "model": MODEL_NAMES,
            "task": TASKS,
            "SWE_threshold_m": SWE_THRESHOLDS,
            "time": spires_fsca.time.data,
            "elev_bin": np.arange(n_elev),
            "aspect_bin": np.arange(n_aspect),
        }
    )
    binned_ds.attrs.update({
        "description": "Performance metrics binned by elevation and aspect.",
        "elev_bins": elev_bins.tolist(),
        "aspect_bins": aspect_bins.tolist(),
    })
    binned_ds.to_netcdf(BINNED_METRICS_FILE)
    print("Performance metrics binned by terrain saved to:", BINNED_METRICS_FILE)

    # --- Save melt-out date outputs ---
    print("\nSaving melt-out date metrics...")

    melt_out_metric_names = [
        "melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days",
        "melt_out_correlation", "melt_out_n_pixels",
    ]
    melt_out_data = {}
    for metric in melt_out_metric_names:
        data = np.full(
            (len(MODEL_NAMES), len(TASKS), len(SWE_THRESHOLDS), len(water_years)), np.nan
        )
        for (model, task, swe, wy), ds in melt_out_metrics.items():
            mi = MODEL_NAMES.index(model)
            ti = TASKS.index(task)
            si = SWE_THRESHOLDS.index(swe)
            wi = list(water_years).index(wy)
            data[mi, ti, si, wi] = ds[metric].item()
        melt_out_data[metric] = (["model", "task", "SWE_threshold_m", "water_year"], data)

    melt_out_metrics_ds = xr.Dataset(
        melt_out_data,
        coords={
            "model": MODEL_NAMES,
            "task": TASKS,
            "SWE_threshold_m": SWE_THRESHOLDS,
            "water_year": water_years,
        },
    )
    melt_out_metrics_ds.attrs.update({
        "description": (
            "Melt-out date comparison metrics (model - SPIReS), in days. "
            "Computed only at pixels where SPIReS observed snow cover "
            f"(fSCA >= {MELT_OUT_FSCA_THRESH}) during that water year, to keep "
            "canopy-obscured SPIReS pixels from biasing the comparison."
        ),
        "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH,
        "water_year_start_month": WATER_YEAR_START_MONTH,
    })
    melt_out_metrics_ds.to_netcdf(MELT_OUT_METRICS_FILE)
    print("Melt-out date metrics saved to:", MELT_OUT_METRICS_FILE)

    # --- Save melt-out terrain-binned metrics (elevation/aspect) ---
    print("Saving melt-out terrain-binned metrics...")
    melt_out_binned_names = ["melt_out_bias_days", "melt_out_MAE_days", "melt_out_RMSE_days", "melt_out_n_pixels"]
    melt_out_binned_data = {}
    for metric in melt_out_binned_names:
        data = np.full(
            (len(MODEL_NAMES), len(TASKS), len(SWE_THRESHOLDS), len(water_years), n_elev, n_aspect), np.nan
        )
        for (model, task, swe, wy), ds in melt_out_binned.items():
            mi = MODEL_NAMES.index(model)
            ti = TASKS.index(task)
            si = SWE_THRESHOLDS.index(swe)
            wi = list(water_years).index(wy)
            if metric in ds:
                data[mi, ti, si, wi, :, :] = ds[metric].values
        melt_out_binned_data[metric] = (
            ["model", "task", "SWE_threshold_m", "water_year", "elev_bin", "aspect_bin"], data
        )

    melt_out_binned_ds = xr.Dataset(
        melt_out_binned_data,
        coords={
            "model": MODEL_NAMES,
            "task": TASKS,
            "SWE_threshold_m": SWE_THRESHOLDS,
            "water_year": water_years,
            "elev_bin": np.arange(n_elev),
            "aspect_bin": np.arange(n_aspect),
        },
    )
    melt_out_binned_ds.attrs.update({
        "description": "Melt-out date errors (model - SPIReS, days) binned by elevation and aspect.",
        "elev_bins": elev_bins.tolist(),
        "aspect_bins": aspect_bins.tolist(),
        "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH,
    })
    melt_out_binned_ds.to_netcdf(MELT_OUT_BINNED_FILE)
    print("Melt-out terrain-binned metrics saved to:", MELT_OUT_BINNED_FILE)

    if SAVE_MELT_OUT_GRIDS:
        print("Saving gridded melt-out dates...")
        y_coord, x_coord = spires_fsca["y"], spires_fsca["x"]
        model_grid_data = np.full(
            (len(MODEL_NAMES), len(TASKS), len(SWE_THRESHOLDS), len(water_years),
             y_coord.size, x_coord.size), np.nan
        )
        for (model, task, swe, wy), doy_da in melt_out_grids.items():
            mi = MODEL_NAMES.index(model)
            ti = TASKS.index(task)
            si = SWE_THRESHOLDS.index(swe)
            wi = list(water_years).index(wy)
            model_grid_data[mi, ti, si, wi, :, :] = doy_da.values

        spires_grid_data = np.stack(
            [spires_melt_out_by_wy[wy].where(spires_snow_mask_by_wy[wy]).values for wy in water_years],
            axis=0,
        )

        melt_out_grids_ds = xr.Dataset(
            {
                "model_melt_out_dowy": (
                    ["model", "task", "SWE_threshold_m", "water_year", "y", "x"],
                    model_grid_data,
                ),
                "SPIReS_melt_out_dowy": (["water_year", "y", "x"], spires_grid_data),
            },
            coords={
                "model": MODEL_NAMES,
                "task": TASKS,
                "SWE_threshold_m": SWE_THRESHOLDS,
                "water_year": water_years,
                "y": y_coord,
                "x": x_coord,
            },
        )
        melt_out_grids_ds.attrs.update({
            "description": (
                "Per-pixel melt-out day-of-water-year (DOWY; 1 = Oct 1), masked to "
                "pixels SPIReS observed as snow-covered that water year."
            ),
            "melt_out_fsca_thresh": MELT_OUT_FSCA_THRESH,
        })
        melt_out_grids_ds.rio.write_crs(CRS).to_netcdf(MELT_OUT_GRIDS_FILE)
        print("Gridded melt-out dates saved to:", MELT_OUT_GRIDS_FILE)

    print("\nDone!")


if __name__ == "__main__":
    main()
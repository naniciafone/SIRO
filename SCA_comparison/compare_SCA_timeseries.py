#!/usr/bin/env python
"""
Compare model- and satellite-derived snow-covered area (SCA) time series

Rainey Aberle (rainey.aberle@usace.army.mil)
USACE-ERDC-CRREL
June 2026
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from glob import glob
from tqdm import tqdm
from rasterio.enums import Resampling
import xrspatial


# ----- CONFIGURATION -----
base_dir = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
model_sca_dir = os.path.join(base_dir, "model_SCA_maps")
model_names = ["HMS-EB", "HMS-TI", "iSnobal", "SnowModel"]
pss_sca_files = sorted(glob(os.path.join(base_dir, "PSS_SCA_maps", "*.tif")))
dem_file = os.path.join(base_dir, "..", "DEMs", "USGS_2026_DEM_merged.tif")
CRS = "EPSG:32611"
out_dir = os.path.join(base_dir, "SCA_comparison_results")
os.makedirs(out_dir, exist_ok=True)


# ----- HELPER FUNCTIONS -----
def sca_and_masked_area(sca_da):
    res = sca_da.rio.resolution()
    pixel_area = abs(res[0] * res[1])
    snow = xr.where(sca_da==1, 1, 0).sum(dim=['x', 'y']).data * pixel_area
    masked = xr.where(np.isnan(sca_da), 1, 0).sum(dim=['x', 'y']).data * pixel_area
    valid = xr.where(~np.isnan(sca_da), 1, 0).sum(dim=['x', 'y']).data * pixel_area
    return snow, masked, valid

def load_and_reproject_pss_sca(
    pss_files: list[str] = None,
    model_file: str = None,
    out_file: str = None,
    chunks: dict = None,
    resampling: str = "average",
    threshold: float = 0.5,
    ) -> xr.DataArray:

    model_name = os.path.basename(model_file).split("_")[0]
    print(f"Reprojecting PlanetScope SCA to {model_name} grid")

    if not pss_files:
        raise FileNotFoundError("pss_files list is empty.")

    if chunks is None:
        chunks = {"y": 1024, "x": 1024}

    # Load existing output
    if os.path.exists(out_file):
        print(f"Output file already exists, loading: {out_file}")
        cube = xr.open_dataset(out_file).SCA
        return cube

    # Build target grid
    with xr.open_dataset(model_file).SCA.squeeze() as model_ds:
        model_ds = model_ds.rio.write_crs("EPSG:32611")
        target_grid = model_ds.isel(time=0)

    # Construct resampling method
    try:
        resampling_method = Resampling[resampling]
    except KeyError:
        valid = ", ".join(Resampling.__members__)
        raise ValueError(
            f"Invalid resampling method '{resampling}'. "
            f"Valid methods are: {valid}"
        )

    reprojected = []
    for filepath in tqdm(sorted(pss_files)):
        # Open raster
        with rxr.open_rasterio(filepath, chunks=chunks).squeeze() as da:

            # Convert nodata -> NaN
            da = xr.where(da == 255, np.nan, da)

            # Assign nodata, datatype, and CRS for rasterio resampling
            da = da.astype("float32")
            da = da.rio.write_nodata(np.nan)
            da = da.rio.write_crs("EPSG:32611")

            # Parse date from filename
            date = os.path.basename(filepath).split("_")[0]
            dt = np.datetime64(date)

            # Add time coordinate
            da = da.assign_coords(time=dt).expand_dims("time")

            # Reproject to model grid
            da = da.rio.reproject_match(
                target_grid,
                resampling=resampling_method,
            )

            # Threshold while preserving NaNs
            da = xr.where(
                np.isnan(da),
                np.nan,
                xr.where(da >= threshold, 1.0, 0.0)
            ).astype("float32")

            reprojected.append(da)

    # Concatenate and organize
    cube = xr.concat(reprojected, dim="time").sortby("time")
    cube = cube.rio.write_crs("EPSG:32611")
    cube.name = "SCA"
    cube.attrs.update({
        "long_name": "Binary snow-covered area",
        "units": "unitless",
        "snow_value": 1.0,
        "no_snow_value": 0.0,
        "model_grid": model_name,
        "fSCA_threshold": threshold,
        "resampling_method": resampling,
    })

    # Save
    cube.to_netcdf(out_file)
    print("Saved regridded PlanetScope time series to:", out_file)

    return cube

def calc_confusion_matrix(model_sca, ref_sca):
    m = model_sca.data.ravel().astype(bool)
    r = ref_sca.data.ravel().astype(bool)
    valid = ~np.isnan(m) & ~np.isnan(r)
    m = m[valid]
    r = r[valid]
    tp = int(np.sum(m & r))
    tn = int(np.sum(~m & ~r))
    fp = int(np.sum(m & ~r))
    fn = int(np.sum(~m & r))
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def get_elev_bins(dem_da, bin_width=50):
    min_elev = np.nanmin(dem_da.data)
    max_elev = np.nanmax(dem_da.data)
    start = bin_width * (min_elev // bin_width)
    end = bin_width * (1 + (max_elev // bin_width))
    return np.arange(start, end + bin_width, bin_width)

def get_aspect_bins(n_bins=8):
    return np.linspace(0, 2 * np.pi, n_bins + 1)

def regrid_to_model(pss_da, model_da):
    pss_da = pss_da.rio.write_crs(CRS)
    model_da = model_da.rio.write_crs(CRS)
    return pss_da.rio.reproject_match(model_da, resampling="average")

def load_dem_for_grid(model_da):
    dem = rxr.open_rasterio(dem_file).squeeze()
    dem = dem.rio.write_crs(CRS)
    dem = dem.rio.reproject_match(model_da)
    return dem

def compute_recall_by_terrain(pss_sca_da, model_sca_da, elev_da, aspect_da, elev_bins, aspect_bins):
    # Bin elevation and aspect
    elev_bin_idx = np.digitize(elev_da.values, elev_bins) - 1
    aspect_bin_idx = np.digitize(aspect_da.values, aspect_bins) - 1

    n_elev_bins = len(elev_bins) - 1
    n_aspect_bins = len(aspect_bins) - 1
    n_time = pss_sca_da.sizes['time']

    recall = np.full((n_time, n_elev_bins, n_aspect_bins), np.nan)

    # Loop over bins and time
    for i in range(n_time):
        sat_sca = pss_sca_da.isel(time=i).values
        mod_sca = model_sca_da.isel(time=i).values

        for j in range(n_elev_bins):
            for k in range(n_aspect_bins):
                mask = (elev_bin_idx == j) & (aspect_bin_idx == k)
                # True positives
                tp = np.sum((sat_sca[mask] == 1) & (mod_sca[mask] == 1))
                # False negatives
                fn = np.sum((sat_sca[mask] == 1) & (mod_sca[mask] == 0))
                # Recall
                denom = tp + fn
                recall[i, j, k] = tp / denom if denom > 0 else np.nan

    # Build Dataset
    recall_da = xr.DataArray(
        recall,
        dims=['time', 'elev_bin', 'aspect_bin'],
        coords={
            'time': pss_sca_da['time'],
            'elev_bin': np.arange(n_elev_bins),
            'aspect_bin': np.arange(n_aspect_bins)
        },
        name='recall'
    )

    recall_ds = xr.Dataset({'recall': recall_da})
    recall_ds['recall'].attrs['elev_bins'] = elev_bins
    recall_ds['recall'].attrs['aspect_bins'] = aspect_bins

    return recall_ds


# ----- MAIN WORKFLOW -----
def main():
    # --- Calculate PlanetScope areas time series at native grid ---
    sca_pss_native_file = os.path.join(out_dir, "SCA_totals_timeseries_PlanetScope_native_grid.csv")
    if os.path.exists(sca_pss_native_file):
        print(f"PlanetScope areas time series at native grid already exists, loading from file: {sca_pss_native_file}")
        sca_pss_native_df = pd.read_csv(sca_pss_native_file)

    else:
        print("Calculating PlanetScope areas time series at native grid...")
        sca_pss_native_list = []
        for pss_sca_file in tqdm(pss_sca_files):
            with rxr.open_rasterio(pss_sca_file, masked=True).squeeze() as pss_sca_da:
                sca, masked, valid = sca_and_masked_area(pss_sca_da)
            df = pd.DataFrame({
                'datetime': [os.path.basename(pss_sca_file).split('_')[0]],
                'SCA_PlanetScope_m2': [sca],
                'masked_area_PlanetScope_m2': [masked],
                'valid_area_PlanetScope_m2': [valid]
            }, index=[0])
            sca_pss_native_list += [df]
        sca_pss_native_df = pd.concat(sca_pss_native_list, ignore_index=True)
        sca_pss_native_df.to_csv(sca_pss_native_file, index=False, header=True)
        print(f"Saved to: {sca_pss_native_file}")

    # --- Define elevation and aspect bins for later ---
    print("\nCalculating elevation and aspect bins from native DEM")
    with rxr.open_rasterio(dem_file, masked=True).squeeze() as dem_da:
        elev_bins = get_elev_bins(dem_da, bin_width=50)
        aspect_bins = np.arange(0, 361, 45)

    # --- Iterate over models ---
    for model in model_names:
        print(f"\nProcessing model: {model}")

        # Get input files
        model_files = sorted(glob(os.path.join(model_sca_dir, f"{model}*.nc")))
        if not model_files:
            print(f"  No files found for {model}")
            continue
        # NOTE: Found no difference in SCA when varying SWE threshold,
        # so just processing the first file for each model
        mf = model_files[0]
        model_name = "HMS" if "HMS" in model else model

        # --- Define output files for this model---
        pss_regrid_file = os.path.join(out_dir, f"PlanetScope_SCA_{model_name}_grid.nc")
        sca_file = os.path.join(out_dir, f"SCA_totals_timeseries_{model}.csv")
        cm_file = os.path.join(out_dir, f"confusion_matrices_{model}.csv")
        recall_terrain_file = os.path.join(out_dir, f"recall_binned_elev_aspect_{model}.nc")

        # --- Regrid PlanetScope to model grid ---
        if not os.path.exists(pss_regrid_file):
            print(f"Regridding PlanetScope SCA time series to {model_name} grid")
            pss_regrid = load_and_reproject_pss_sca(pss_sca_files, mf, pss_regrid_file)
        else:
            print("Regridded PlanetScope SCA time series already exists, skipping.")

        # --- SCA totals ---
        if os.path.exists(sca_file):
            print(f"SCA totals CSV exists, skipping: {sca_file}")
        else:
            with (
                xr.open_dataset(mf).SCA.squeeze() as model_sca,
                xr.open_dataset(pss_regrid_file).SCA.squeeze() as pss_regrid
                ):
                # Set no data values to NaN
                model_sca = xr.where(model_sca == 255, np.nan, model_sca)
                pss_regrid = xr.where(pss_regrid == 255, np.nan, pss_regrid)

                # Model areas time series
                model_sca = model_sca.rio.write_crs(CRS)
                model_sca = model_sca.sel(time=pss_regrid["time"], method="nearest")
                model_snow, model_masked, model_valid = sca_and_masked_area(model_sca)

                # PlanetScope regridded areas time series
                pss_regrid_snow, pss_regrid_masked, pss_regrid_valid = sca_and_masked_area(pss_regrid)

                # Compile in DataFrame
                sca_df = pd.DataFrame({
                    "datetime": pss_regrid.time.values,
                    f"SCA_{model}_m2": model_snow,
                    f"masked_area_{model}_m2": model_masked,
                    f"valid_area_{model}_m2": model_valid,
                    f"SCA_PSS-{model}-regrid_m2": pss_regrid_snow,
                    f"masked_area_PSS-{model}-regrid_m2": pss_regrid_masked,
                    f"valid_area_PSS-{model}-regrid_m2": pss_regrid_valid,
                })
                # round values cuz we're not that precise
                sca_df.iloc[:, 2:] = sca_df.iloc[:, 2:].round()
                sca_df.to_csv(sca_file, index=False, header=True)
                print("SCA totals saved to:", sca_file)

        # --- Confusion matrix ---
        if os.path.exists(cm_file):
            print("Confusion matrix time series exists, skipping:", cm_file)
        else:
            print("Calculating confusion matrix time series...")
            with (
                xr.open_dataset(mf).SCA.squeeze() as model_sca,
                xr.open_dataset(pss_regrid_file).SCA.squeeze() as pss_regrid
                ):
                model_sca = model_sca.rio.write_crs(CRS)
                model_sca = model_sca.sel(time=pss_regrid["time"], method="nearest")
                cm_list = []
                for i in range(len(model_sca["time"])):
                    cm = calc_confusion_matrix(model_sca.isel(time=i), pss_regrid.isel(time=i))
                    df = pd.DataFrame(cm, index=[0])
                    df["datetime"] = pss_regrid.time.data[i]
                    cm_list += [df]
                cm_df = pd.concat(cm_list, ignore_index=True)
                cm_df.to_csv(cm_file, index=False, header=True)
                print("Confusion matrix time series saved to:", cm_file)

        # --- Elevation/aspect binned SCA ---
        if os.path.exists(recall_terrain_file):
            print("Recall with binned terrain exists, skipping:", recall_terrain_file)
        else:
            print("Calculating recall with binned terrain...")
            with (
                xr.open_dataset(mf).SCA.squeeze() as model_sca,
                xr.open_dataset(pss_regrid_file).SCA.squeeze() as pss_regrid
                ):
                # Sample model at PSS times
                model_sca = model_sca.rio.write_crs(CRS)
                model_sca = model_sca.sel(time=pss_regrid["time"], method="nearest")

                # Reproject DEM to model grid
                with rxr.open_rasterio(dem_file, masked=True).squeeze() as dem_da:
                    dem_regrid_da = dem_da.rio.reproject_match(model_sca)

                # Calculate aspect from regridded DEM
                aspect_da = xrspatial.aspect(dem_regrid_da)
                
                # Calculate recall with elevation and aspect bins
                recall_terrain_da = compute_recall_by_terrain(pss_regrid, model_sca, dem_regrid_da, aspect_da, elev_bins, aspect_bins)

                # Save to file
                recall_terrain_da.to_netcdf(recall_terrain_file)
                print("Recall with binned terrain saved to:", recall_terrain_file)


    # --- Compile SCA time series ---
    print("\nCompiling SCA totals CSVs...")
    sca_compiled_file = os.path.join(out_dir, "SCA_totals_timeseries_compiled.csv")
    if not os.path.exists(sca_compiled_file):

        # Get SCA time series CSVs
        sca_files = sorted(glob(os.path.join(out_dir, "SCA*.csv")))
        print(len(sca_files))
        
        # Loop over files
        sca_df = None
        for i, file in enumerate(sca_files):
            df = pd.read_csv(file)

            if type(sca_df)!=pd.DataFrame:
                sca_df = df
            else:
                sca_df = pd.merge(sca_df, df, on='datetime')

        # Save to file
        sca_df = sca_df.sort_index(axis=1)
        sca_df = sca_df.set_index('datetime')
        sca_df.to_csv(sca_compiled_file, index=True)
        print("Compiled SCA time series saved to:", sca_compiled_file)

    else:
        print("Compiled SCA time series already exists, skipping.")

if __name__ == "__main__":
    main()
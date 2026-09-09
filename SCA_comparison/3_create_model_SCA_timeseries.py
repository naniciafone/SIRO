#!/usr/bin/env python

"""
Create snow-covered area (SCA) time series for HEC-HMS, iSnobal, and SnowModel.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL
May 2026
"""

import os
from glob import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt

# --- Base config ---
# Inputs
base_dir = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
task = 2
hms_eb_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "HMS_snow_grids", f"Task{task}", "EB", "swe_tif", "*.tiff")))
hms_ti_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "HMS_snow_grids", f"Task{task}", "TI", "swe_tif", "*.tiff")))
is_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "iSnobal", f"m3w_isnobal_task{task}_SWE", "wy*", "mores*.tif")))
sm_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "SnowModel", f"*Task{task}.nc")))
aoi_file = os.path.join(base_dir, "..", "MCS_outline.gpkg")

# Outputs
out_dir = os.path.join(base_dir, "model_SCA")

# Define range of SWE thresholds for determining "snow-covered" vs "snow-free"
swe_thresholds = np.arange(0, 0.051, 0.01) # m (0:10:50 mm)

# --- Processing function ---
def swe_to_sca(files=None, swe_thresh=0.01, out_file=None, clip_gdf=None):
    # Create compiled SCA dataset
    print("Compiling SWE time series, calculating SCA...")
    sca_list = []
    for file in tqdm(files):
        # Open dataset
        with xr.open_dataset(file).squeeze() as swe:
            
            swe = swe.rio.write_crs("EPSG:32611")

            # Clip to clip_gdf
            if type(clip_gdf)==gpd.GeoDataFrame:
                swe = swe.rio.clip(clip_gdf.geometry, crs="EPSG:32611")

            # Calculate SCA, maintaining NaN values
            sca = xr.where(np.isnan(swe), np.nan, swe > swe_thresh)

            # For HMS and iSnobal, parse date from the file name and add the time stamp
            if "HMS" in os.path.basename(out_file):
                date = os.path.basename(file).split("T0000_")[1].replace('_','-')
                dt = np.datetime64(date)
                sca = sca.expand_dims(time=[dt])
            elif "iSnobal" in os.path.basename(out_file):
                if task==1:
                    date = os.path.splitext(os.path.basename(file))[0].split("mores_creek_SWE_")[1]
                else:
                    date = os.path.splitext(os.path.basename(file))[0].split("mores_creek_swe_")[1]
                date = date[0:13] + ':' + date[14:16] + ':' + date[17:]
                dt = np.datetime64(date)
                sca = sca.expand_dims(time=[dt])

            sca_list += [sca]
    
    # Concatenate datasets
    sca_compiled = xr.concat(sca_list, dim='time', data_vars='all')

    # Convert to integer with nodata value=255
    sca_compiled = xr.where(np.isnan(sca_compiled), 255, sca_compiled).astype(np.uint8)

    # Reduce to DataArray named "SCA"
    sca_da = sca_compiled[list(sca_compiled.data_vars)[0]].rename("SCA").squeeze()
    
    # Set attributes
    sca_da.attrs.update({
        "long_name": "Binary snow-covered area",
        "units": "unitless",
        "snow_value": 1,
        "no_snow_value": 0,
        "SWE_threshold_m": swe_thresh,
        "nodata": 255,
    })
    sca_da.rio.write_nodata(255)
    sca_da = sca_da.rio.write_crs("EPSG:32611")

    # Save to file
    sca_da.to_netcdf(out_file)
    print("Compiled SCA time series saved to:", out_file)

    return sca_da


# --- Main workflow ---
def main():

    print(f"Located {len(hms_eb_swe_files) + len(hms_ti_swe_files) + len(sm_swe_files) + len(is_swe_files)} total model SWE files for Task {task}.")
    print(f"\tHMS EB: {len(hms_eb_swe_files)}")
    print(f"\tHMS TI: {len(hms_ti_swe_files)}")
    print(f"\tiSnobal: {len(is_swe_files)}")
    print(f"\tSnowModel: {len(sm_swe_files)}")

    os.makedirs(out_dir, exist_ok=True)

    # Load the AOI for clipping
    aoi_gdf = gpd.read_file(aoi_file)

    # Iterate over models
    for swe_files, model_name in zip(
        [hms_eb_swe_files, hms_ti_swe_files, is_swe_files, sm_swe_files],
        ["HMS-EB", "HMS-TI", "iSnobal", "SnowModel"]
        ):
        print(f"\n{model_name}\n----------")

        if len(swe_files) < 1:
            print("No SWE files found, skipping.")
            continue

        sca_das = []
        thresholds_used = []

        # Iterate over SWE thresholds
        for swe_threshold in swe_thresholds:
            print(f"SWE threshold = {swe_threshold} m")

            # Define output file
            sca_file = os.path.join(out_dir, f"{model_name}_SCA_timeseries_Task{task}_SWEthresh{swe_threshold}m.nc")

            # Check if it already exists
            if not os.path.exists(sca_file):
                # Create compiled SCA time series from SWE
                swe_to_sca(
                    swe_files, 
                    out_file = sca_file,
                    swe_thresh=swe_threshold,
                    clip_gdf=aoi_gdf
                )

            # Load SCA time series
            sca_da = xr.open_dataset(sca_file).SCA.squeeze()
            # Ensure time is sorted
            sca_da = sca_da.sortby('time')
            # Add threshold as a new coordinate
            sca_da = sca_da.expand_dims(threshold=[swe_threshold])
            sca_das.append(sca_da)
            thresholds_used.append(swe_threshold)

        # Merge all thresholds into one dataset
        sca_all = xr.concat(sca_das, dim='threshold')

        # Calculate total SCA (number of snow-covered pixels) for each time and threshold
        # SCA==1 is snow, 0 is no snow, 255 is nodata
        res = (sca_all.x.data[1] - sca_all.x.data[0])**2 # m2
        total_sca = (sca_all == 1).sum(dim=['x', 'y']) * res

        # Plot
        plt.figure(figsize=(10,6))
        colors = [plt.cm.viridis(i/len(thresholds_used)) for i in range(len(thresholds_used))]
        for i, swe_threshold in enumerate(thresholds_used):
            plt.plot(
                sca_all.time.values,
                total_sca.sel(threshold=swe_threshold).values / 1e6,
                label=f"SWE ≥ {swe_threshold:.2f} m",
                color=colors[i]
            )
        plt.title(f"Total SCA Time Series for {model_name}")
        plt.xlabel("Date")
        plt.ylabel("SCA (km$^2$)")
        plt.legend(title="SWE Threshold")
        plt.tight_layout()
        fig_file = os.path.join(out_dir, f"{model_name}_SCA_vs_threshold.png")
        plt.savefig(fig_file, dpi=250, bbox_inches='tight')
        print("SCA vs. threshold figure saved to:", fig_file)

    print("\nDone! :3\n")

if __name__=="__main__":
    main()
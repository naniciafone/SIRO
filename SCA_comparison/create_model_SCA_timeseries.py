#!/usr/bin/env python

"""
Create snow-covered area (SCA) time series for HEC-HMS, iSnobal, and SnowModel

Rainey Aberle (rainey.aberle@usace.army.mil)
USACE-ERDC-CRREL
May 2026
"""

import os
from glob import glob
import xarray as xr
import numpy as np
from tqdm import tqdm
import geopandas as gpd

# --- Base config ---
# Inputs
base_dir = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
hms_eb_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "HMS_snow_grids", "EB", "swe_tif", "*.tiff")))
hms_ti_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "HMS_snow_grids", "TI", "swe_tif", "*.tiff")))
is_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "iSnobal", "m3w_isnobal_task1_SWE", "wy*", "mores*.tif")))
sm_swe_files = sorted(glob(os.path.join(base_dir, "model_SWE", "SnowModel", "*.nc")))
aoi_file = os.path.join(base_dir, "..", "MCS_outline.gpkg")

# Outputs
out_dir = os.path.join(base_dir, "model_SCA_maps")

# Define range of SWE thresholds for determining "snow-covered" vs "snow-free"
swe_thresholds = [0.000, 0.005, 0.01, 0.015, 0.020] # m

# --- Processing function ---
def swe_to_sca(files=None, swe_thresh=0.05, out_file=None, clip_gdf=None):
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
            sca = xr.where(np.isnan(swe), np.nan, swe >= swe_thresh)

            # For HMS and iSnobal, parse date from the file name and add the time stamp
            if "HMS" in os.path.basename(out_file):
                date = os.path.basename(file).split("T0000_")[1].replace('_','-')
                dt = np.datetime64(date)
                sca = sca.expand_dims(time=[dt])
            elif "iSnobal" in os.path.basename(out_file):
                date = os.path.splitext(os.path.basename(file))[0].split("mores_creek_SWE_")[1]
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
    sca_da = sca_da.rio.write_crs("EPSG:32611")

    # Save to file
    sca_da.to_netcdf(out_file)
    print("Compiled SCA time series saved to:", out_file)

    return sca_da


# --- Main workflow ---
def main():
    print(f"Located {len(hms_eb_swe_files) + len(hms_ti_swe_files) + len(sm_swe_files) + len(is_swe_files)} total model SWE files.")
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

        # Iterate over SWE thresholds
        for swe_threshold in swe_thresholds:
            print(f"SWE threshold = {swe_threshold} m")

            # Define output file
            sca_file = os.path.join(out_dir, f"{model_name}_SCA_timeseries_SWEthresh{swe_threshold}m.nc")

            # Check if it already exists
            if os.path.exists(sca_file):
                print(f"SCA time series already exists, skipping.")
            else:

                # Create compiled SCA time series from SWE
                sca_compiled = swe_to_sca(
                    swe_files, 
                    out_file = sca_file,
                    clip_gdf=aoi_gdf
                    )
                

    print("\nDone! :3\n")

if __name__=="__main__":
    main()
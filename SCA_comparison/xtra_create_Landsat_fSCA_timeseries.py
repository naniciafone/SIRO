#!/usr/bin/env python

# Compile fractional snow-covered area (fSCA) products from Landsat Collection 2

import os
from glob import glob
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import numpy as np
from tqdm import tqdm


# I/O
BASE_DIR = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/"
LS_FSCA_FILES = sorted(glob(os.path.join(BASE_DIR, "Landsat_fSCA", "*GROUND_SNOW.TIF")))
AOI_FILE = os.path.join(BASE_DIR, "..", "MCS_outline.gpkg")
OUT_FILE = os.path.join(BASE_DIR, "SCA_comparison_results_SPIReS", "Landsat_fSCA.nc")
print(f"Identified {len(LS_FSCA_FILES)} Landsat fSCA files.")

CRS = "EPSG:32611"
THREADS = 8

# Helper function
def process_ls_fsca(file, aoi):
    # Clip to AOI
    with xr.open_dataset(file, mask_and_scale=True).squeeze() as ds:
        ds = ds.rio.reproject(aoi.crs)
        ds = ds.rio.clip(aoi.geometry)

    # Add time dimension
    date = os.path.basename(file).split('_')[3]
    dt = np.datetime64(f"{date[0:4]}-{date[4:6]}-{date[6:]}")
    ds = ds.expand_dims(time=[dt])

    # Divide by ten so fSCA ranges from 0-100
    ds /= 10

    return ds

# Main processing function
def main():
    aoi = gpd.read_file(AOI_FILE)
    aoi = aoi.to_crs(CRS)

    # Process files
    ds_list = []
    for file in tqdm(LS_FSCA_FILES, desc="Processing files:"):
        ds = process_ls_fsca(file, aoi)
        ds_list += [ds]

    ds_full = xr.concat(ds_list, dim='time')
    ds_full = ds_full.sortby('time')

    # Save to file
    ds_full.to_netcdf(OUT_FILE)
    print("Clipped, compiled dataset saved to:", OUT_FILE)



if __name__=="__main__":
    main()
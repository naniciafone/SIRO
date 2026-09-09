# !/usr/bin/env python

"""
Create fractional snow-covered area (fSCA) time series reprojected to UTM 
and cropped to the Mores Creek basin from the SPIReS Snow Today product.

Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL
July 2026
"""


import os
import xarray as xr
import rioxarray as rxr
from glob import glob
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_map

# I/O
BASE_DIR = "/Users/rdcrlrka/Research/SIRO/MCS_SCA"
OUT_FILE = os.path.join(BASE_DIR, "SCA_comparison_results_SPIReS", "SPIReS_fSCA.nc")
AOI_FILE = os.path.join(BASE_DIR, "..", "MCS_outline.gpkg")

# Number of threads to use for parallel file processing
NUM_THREADS = 5

# Locate all the files
spires_files = sorted(glob(os.path.join(BASE_DIR, "SnowToday", "*.nc")))
print(f"Located {len(spires_files)} SPIReS files.")

# Helper function for processing each file
def process_spires(file):
    # Load the dataset
    with xr.open_dataset(file, mask_and_scale=True)['snow_fraction'].rio.write_crs("ESRI:54008") as fsca:
        # Reproject to UTM
        fsca_utm = fsca.rio.reproject("EPSG:32611")

    # Crop to AOI
    aoi = gpd.read_file(AOI_FILE)
    fsca_utm_crop = fsca_utm.rio.clip(aoi.geometry)

    fsca_utm_crop = fsca_utm_crop.rio.write_crs("EPSG:32611")
    fsca_utm_crop = fsca_utm_crop.rio.write_nodata(np.nan)

    return fsca_utm_crop

# Process files in parallel
fsca_list = p_map(process_spires, spires_files, num_cpus=NUM_THREADS)

# Concatenate
fsca = xr.concat(fsca_list, dim='time')

# Convert to uint16
fsca_int = xr.where(np.isnan(fsca), 255, fsca).astype(int)
fsca_int = fsca_int.rio.write_crs("EPSG:32611")
fsca_int = fsca_int.rio.write_nodata(255)

# Save to file
fsca_int.to_dataset().to_netcdf(OUT_FILE)
print("SPIReS fSCA dataset saved to:", OUT_FILE)

# Plot spatially-averaged fSCA
fsca_mean = fsca.mean(dim=['x', 'y'])
fig, ax = plt.subplots(1, 1, figsize=(8,6))
ax.plot(fsca_mean.time, fsca_mean, '-k')
ax.grid(alpha=0.2)
ax.set_xlabel('Time')
ax.set_ylabel('Spatially averaged fSCA [%]')
fig_file = os.path.splitext(OUT_FILE)[0] + '.png'
fig.savefig(fig_file, dpi=250, bbox_inches='tight')
print("SCA time series figure saved to:", fig_file)
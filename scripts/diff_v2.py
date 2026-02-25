#!/usr/bin/env python
# coding: utf-8

#import
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import fiona
import rasterio
import rasterio.mask
import geopandas as gpd
from datetime import datetime
from rasterio.warp import reproject, Resampling



# set up directories
dir = "."
Task_number = 2

if Task_number == 1:
    out_dir = os.path.join(dir, "outputs/task1/")
else:
    out_dir = os.path.join(dir, "outputs/task2/")

rasters_dir = os.path.join(out_dir, "rasters/")

resample_rasters = {
    "HMS Energy Balance": os.path.join(rasters_dir, "HMS_Energy_Balance_lidar_resample.tif"),
    "HMS Temperature Index": os.path.join(rasters_dir, "HMS_Temperature_Index_lidar_resample.tif"),
    "iSnobal": os.path.join(rasters_dir, "iSnobal_lidar_resample.tif"),
    "SnowModel": os.path.join(rasters_dir, "SnowModel_lidar_resample.tif")}

lidar = os.path.join(rasters_dir, "LiDAR_MCS_clip.tif")
with rasterio.open(lidar) as src:
    lidar_data = src.read(1, masked=True)
    profile = src.profile                 # for writing outputs


for model, raster in resample_rasters.items():
    with rasterio.open(raster) as src:
        resample_data = src.read(1, masked =True)

        diff_data = lidar_data - resample_data

        out_profile = profile.copy()
        out_profile.update(dtype=rasterio.float32, compress="lzw", nodata = -9999)
        out_path_diff = os.path.join(rasters_dir, f"{model.replace(' ', '_')}_lidar_diff.tif")

        with rasterio.open(out_path_diff, "w", **out_profile) as dst:
            dst.write(diff_data.filled(np.nan).astype(np.float32), 1)








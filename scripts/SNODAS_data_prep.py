import rasterio
import requests
import os
import glob
import fiona
import numpy as np
from rasterio.mask import mask
import re

dir= "."
raster_files = glob.glob(os.path.join(dir,"resampled", '*.tif'))


# Clip to Mores Creek basin outline
MCS = os.path.join(dir, "MCS_outline/basin_outline.shp")
with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


##########################
out_dir = os.path.join(dir, "outputs")
for raster in raster_files:
    with rasterio.open(raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=-9999)
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=-9999)

        # Include model in output filename
        date = re.search(r"\d{8}", raster).group()
        out_name = f"SNODAS_{date}_basin_clip.tif"
        out_path = os.path.join(out_dir, out_name)

        with rasterio.open(out_path, "w", **profile) as dest:
            dest.write(out_image)

##########################
#Clip to LiDAR domain
MCS = os.path.join(dir, "MCS_outline/MCS_outline.shp")
with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

rasters = glob.glob(os.path.join(out_dir, '*.tif'))

for raster in rasters:
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

        # Include model in output filename
        out_name = os.path.basename(raster).replace("basin_clip.tif", "MCS.tif")
        out_path = os.path.join(out_dir, out_name)

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
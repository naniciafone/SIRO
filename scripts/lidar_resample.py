# #import
# from os import mkdir
#
import rasterio as rio
import os
import glob
import numpy as np
from rasterio.fill import fillnodata
from osgeo import gdal

LiDAR_dir = "C:/Users/RDCRLSMC/Desktop/SIRO/LiDAR/test/outputs"
method = "cubic"



#Set up directories
threshold_dir = os.path.join(LiDAR_dir, "threshold")
smooth_dir = os.path.join(LiDAR_dir, "smooth")
resampled_dir = os.path.join(LiDAR_dir, "resampled")
abs_dir = os.path.join(LiDAR_dir, "abs")

os.makedirs(threshold_dir, exist_ok=True)
os.makedirs(smooth_dir, exist_ok=True)
os.makedirs(resampled_dir, exist_ok=True)
os.makedirs(abs_dir, exist_ok=True)


raster_files = glob.glob(os.path.join(LiDAR_dir, '*.tif'))


for raster in raster_files:
    with rio.open(raster) as src:
        out_raster = os.path.join(threshold_dir, os.path.basename(raster).replace(".tif", "_filtered.tif"))

        sd = src.read(1, masked=True).filled(np.nan)
        sd[sd < -0.2] = np.nan
        sd[sd > 10] = np.nan

        # 4. Update metadata for NaNs
        meta = src.profile

        # 5. Write filled raster
        with rio.open(out_raster, "w", **meta) as dst:
            dst.write(sd.astype("float32"), 1)


for raster in glob.glob(os.path.join(threshold_dir, "*.tif")):
    with rio.open(raster) as src:
        out_raster = os.path.join(smooth_dir, os.path.basename(raster).replace("_filtered.tif", "_smooth.tif"))

        profile = src.profile
        image = src.read(1, masked=True)

        filled_data = fillnodata(image, max_search_distance=20,smoothing_iterations=3)

        # 4. Update metadata for NaNs
        meta = src.profile.copy()
        meta.update(dtype="float32", nodata=np.nan)

        # 5. Write filled raster
        with rio.open(out_raster, "w", **meta) as dst:
            dst.write(filled_data.astype("float32"), 1)


for raster in glob.glob(os.path.join(smooth_dir, "*.tif")):
    out_raster = os.path.join(resampled_dir, os.path.basename(raster).replace("_filtered.tif", f"_{method}.tif"))

    gdal.Warp(
            out_raster,
            raster,
            dstNodata=float("nan"),
            xRes=100,
            yRes=100,
            resampleAlg=method
            # outputBounds=extent
        )

for raster in glob.glob(os.path.join(resampled_dir, '*.tif')):
    with rio.open(raster) as src:

        out_raster = os.path.join(abs_dir, os.path.basename(raster).replace(".tif", "_abs.tif"))
        sd = src.read(1, masked=True).filled(np.nan)
        sd[sd < 0] = 0
        meta = src.profile
        with rio.open(out_raster, "w", **meta) as dst:
            dst.write(sd.astype("float32"), 1)
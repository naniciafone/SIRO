import os
import glob

import rasterio
from osgeo import gdal
import rasterio as rio
import numpy as np

## LiDAR preparation has 3 steps:
    # 1. Filter out values less than -0.2 and greater than 10 meters
    # 2. Downsample data iteratively: 0.5 (native) -> 1.0 -> 10 -> 100 (m)
    # 3. Set all remaining negatives to 0

#Establish directory with LiDAR files to be downsampled:

dir = "C:/Users/RDCRLSMC/Desktop/SIRO/LiDAR"
filtered = os.path.join(dir, "processed/filtered")
os.makedirs(filtered, exist_ok = True)
resampled = os.path.join(dir, "processed/resampled")
os.makedirs(resampled, exist_ok = True)
final = os.path.join(dir, "processed/final")
os.makedirs(final, exist_ok = True)

# Step 1
raster_files = glob.glob(os.path.join(dir, '*.tif'))

for raster in raster_files:
    with rio.open(raster) as src:
        outraster = os.path.join(filtered, os.path.basename(raster))
        sd = src.read(1, masked=True).filled(np.nan)
        sd[sd < -0.2] = np.nan
        sd[sd > 10] = np.nan

        meta = src.profile.copy()
        meta.update(dtype="float32", nodata=np.nan)

        with rio.open(outraster, "w", **meta) as dst:
            dst.write(sd.astype("float32"), 1)


filtered_rasters = glob.glob(os.path.join(filtered, '*.tif'))

for raster in filtered_rasters:
    out_raster = os.path.join(resampled, os.path.basename(raster))
    ds_1 = gdal.Warp(
"",
            raster,
            dstNodata=float("nan"),
            #format="MEM",
            xRes=1,
            yRes=1,
            resampleAlg="cubic")

    ds_10 = gdal.Warp(
"",
            ds_1,
            dstNodata=float("nan"),
            #format="MEM",
            xRes=10,
            yRes=10,
            resampleAlg="cubic")

    ds_100 = gdal.Warp(
            out_raster,
            ds_10,
            dstNodata=float("nan"),
            xRes=100,
            yRes=100,
            resampleAlg="cubic")


resampled_rasters = glob.glob(os.path.join(resampled, '*.tif'))

for raster in resampled_rasters:
    with rasterio.open(raster) as src:
        outraster = os.path.join(final, os.path.basename(raster))
        sd = src.read(1, masked=True).filled(np.nan)
        sd[sd < 0] = 0

    meta = src.profile.copy()
    meta.update(dtype="float32", nodata=np.nan)

    with rio.open(outraster, "w", **meta) as dst:
        dst.write(sd.astype("float32"), 1)

#%%
import glob
from osgeo import gdal
import os

in_dir = "directory here"
raster_files = glob.glob(os.path.join(in_dir, '*.tif'))

res = 100

extent = (601558.000, 4862467.500, 609431.500, 4870872.500)
for raster in raster_files:
    name, ext = os.path.splitext(os.path.basename(raster))
    out_raster = os.path.join(
        in_dir,
        f"{name}_{res}m{ext}"
    )

    gdal.Warp(
        out_raster,
        raster,
        dstNodata=float("nan"),
        xRes=res,
        yRes=res,
        resampleAlg="cubic",
        outputBounds=extent)
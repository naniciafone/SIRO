from osgeo import gdal, osr
import glob
import os

## assuming we have a directory of SNODAS files converted to geotiff and correctly projected
##Given the size of the SNODAS files, I had the must success when roughly clipping the files to the AOI
##This way, I only needed to resample my basin, rather than all of CONUS

dir = "."
out_dir = os.path.join(dir, "raw_clipped")
os.makedirs(out_dir, exist_ok=True)

tif_files = glob.glob(os.path.join(dir, '*.tif'))
output_bounds = [572490.0, 4831530.0, 610000.0, 4877940.0]
output_srs = 'EPSG:32611'

for f in tif_files:
    f_out = os.path.join(out_dir, os.path.basename(f))
    dst_ds = gdal.Warp(
        f_out,                      # Output file path
        f,                       # Input file path
        outputBounds=output_bounds, # Clipping extent
        dstSRS=output_srs,          # SRS of the clipping extent
        dstNodata=-9999,            # Set NoData in the output file to -9999
        resampleAlg='bilinear',     # Resampling algorithm
    )

    dst_ds = None
        
        

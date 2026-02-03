import rasterio
from rasterio.enums import Resampling
import os
import glob

dir = "."  
target_resolution = 10
raster_files = glob.glob(os.path.join(dir, '*.tif'))

for raster in raster_files:

    with rasterio.open(raster) as src:
        # Calculate scale factor
        scale_factor_x = src.res[0] / target_resolution
        scale_factor_y = src.res[1] / target_resolution

        # Calculate new height and width (must be integers)
        new_height = int(src.height * scale_factor_y)
        new_width = int(src.width * scale_factor_x)

        # Read the data with the new shape and resampling method
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear # Or another method like Resampling.nearest, etc.
        )

        # Update the affine transform for the new resolution
        transform = src.transform * src.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )

        # Update the profile for the output file
        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": transform,
            "nodata": src.nodata # Ensure NoData value is carried over
        })

        out_path = os.path.join(dir, os.path.basename(raster).replace(".tif", "_100m.tif"))

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

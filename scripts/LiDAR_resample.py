import rasterio
import numpy as np
from pathlib import Path

# -----------------------
# USER SETTINGS
# -----------------------
src_dir = Path(
    "C:/Users/RDCRLSMC/Desktop/IDAHO_ALS/SNEX_MCS_Lidar_1-20250512_221456/SD"
)
dst_dir = src_dir / "aggregated_100m"
dst_dir.mkdir(exist_ok=True)

scale = 200              # 0.5 m → 100 m
min_coverage = 0.10      # require 10% valid pixels per block

# -----------------------
# PROCESS FILES
# -----------------------
for src_path in src_dir.glob("*.tif"):
    dst_path = dst_dir / f"{src_path.stem}_100{src_path.suffix}"

    with rasterio.open(src_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform

    # Ensure float for NaN handling
    data = data.astype("float32")

    # -----------------------
    # Trim edges to be divisible by scale
    # -----------------------
    h, w = data.shape
    # h = height (# of rows), w = width (number of columns) of reference raster
    # determine how many rows/ columns to keep when rescaling, where h2 and w2 are width and height of output raster
    # i.e., make width and height of output dataset multiple of scale (200)
    h2 = h - (h % scale)
    # 'h % scale' returns the remainder when h is divided by scale
    # could also use, h2 = (h // scale) * scale
    w2 = w - (w % scale)
    # 'w % scale' returns the remainder when w is divided by scale
    # could also use, w2 = (w // scale) * scale
    data = data[:h2, :w2]

    # -----------------------
    # Reshape into blocks
    # -----------------------
    blocks = data.reshape(
        h2 // scale, scale,
        w2 // scale, scale
    )
    # reshape takes a 2D array (h2, w2) and splits it into 4 dimensions:
        # blocks.shape == (coarse_rows, fine_rows, coarse_cols, fine_cols)
    # where course is the size of the resampled pixel (100-m), and fine is the original pixels that will be used to compute the value of the block
    # reshape does no copying, just groups pixels into blocks, ready for aggregation.
    # -----------------------

    # Aggregate
    # -----------------------
    mean = np.nanmean(blocks, axis=(1, 3))
    # axis relative to blocks.shape (coarse_rows, fine_rows, coarse_cols, fine_cols)
    # so, averaging across pixels in the block

    coverage = (
        np.isfinite(blocks).sum(axis=(1, 3))
    # returns a boolean array of the same shape, True is pixel is a number, false if pixel is NAN
    # so, ensures the sum of the fine rows is not 0
    #determines how many of the pixels (percentage) in a block are NAN
        / (scale * scale)
    )

    #if coverage is less than 10%, make block NAN
    mean[coverage < min_coverage] = np.nan

    # -----------------------
    # Update georeferencing
    # -----------------------
    new_transform = transform * rasterio.Affine.scale(scale, scale)

    profile.update(
        height=mean.shape[0],
        width=mean.shape[1],
        transform=new_transform,
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        BIGTIFF="YES"
    )

    # -----------------------
    # Write output
    # -----------------------
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(mean, 1)

    print(f"Aggregated → {dst_path.name}")

print("Done.")

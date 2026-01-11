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
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    data = data[:h2, :w2]

    # -----------------------
    # Reshape into blocks
    # -----------------------
    blocks = data.reshape(
        h2 // scale, scale,
        w2 // scale, scale
    )

    # -----------------------
    # Aggregate
    # -----------------------
    mean = np.nanmean(blocks, axis=(1, 3))

    coverage = (
        np.isfinite(blocks).sum(axis=(1, 3))
        / (scale * scale)
    )

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

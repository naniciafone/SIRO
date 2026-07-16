#!/usr/bin/env python
# coding: utf-8

#import
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import fiona
import rasterio.mask
import geopandas as gpd
from datetime import datetime
from rasterio.warp import reproject, Resampling



# set up directories
dir = "."
Task_number = 2

if Task_number == 1:
    modeled = os.path.join(dir, "modeled/Task1")
else:
    modeled = os.path.join(dir, "modeled/Task2")

if Task_number == 1:
    out_dir = os.path.join(dir, "outputs/task1/")
else:
    out_dir = os.path.join(dir, "outputs/task2/")

rasters_dir = os.path.join(out_dir, "rasters/")
figs_dir = os.path.join(out_dir, "figs/")

#call lidar
lidar = glob.glob(os.path.join(dir, "lidar","*SD.tif"))
lidar_2000 = glob.glob(os.path.join(dir, "lidar","*2000*"))
if not lidar:
    raise FileNotFoundError("No SD lidar raster found")

if not lidar_2000:
    raise FileNotFoundError("No 2000 m lidar raster found")

# Split by underscore and pick the part that looks like a date
parts = os.path.basename(lidar[0]).split("_")
for part in parts:
    if part.isdigit() and len(part) == 8:
        date_str = part
        break

date_obj = datetime.strptime(date_str, "%Y%m%d")
outfile_100 = os.path.join(rasters_dir, "lidar_sd_"+date_str+".tif")

if os.path.exists(outfile_100):
    print(f"Skipping existing LiDAR raster: {outfile_100}")
    lidar_raster = outfile_100
else:
    with rasterio.open(lidar[0]) as src:
        data = src.read(1, masked=True)
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=-9999)

    #Replace NaNs with -9999
    data_filled = np.where(np.isnan(data), -9999, data)

    # Write the new raster
    with rasterio.open(outfile_100, "w", **profile) as dst:
        dst.write(data_filled.astype("float32"), 1)
        
    lidar_raster = outfile_100

#Do the same for the 2-km LiDAR raster
    
outfile_2000 = os.path.join(rasters_dir, "lidar_sd_"+date_str+"_2000.tif")

if os.path.exists(outfile_2000):
    print(f"Skipping existing LiDAR raster: {outfile_2000}")
    lidar_raster_2000 = outfile_2000
else:
    with rasterio.open(lidar_2000[0]) as src:
        data = src.read(1, masked=True)
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=-9999)

    #Replace NaNs with -9999
    data_filled = np.where(np.isnan(data), -9999, data)

    # Write the new raster
    with rasterio.open(outfile_2000, "w", **profile) as dst:
        dst.write(data_filled.astype("float32"), 1)
        
    lidar_raster_2000 = outfile_2000

HMS_EB = glob.glob(os.path.join(modeled, "*eb_snow*.tiff"))[0]

HMS_TI = glob.glob(os.path.join(modeled, "*TI_snow*.tif"))[0]

out_path = os.path.join(modeled, "HMS_EB_inches.tif")
if os.path.exists(out_path):
    print(f"Skipping existing raster: {out_path}")
else:
    with rasterio.open(HMS_EB) as src:
        raster_data = src.read(1, masked=True).filled(np.nan)
        out_raster = raster_data * 0.0254
        
        # Replace NaN with NoData value
        nodata_val = -9999
        out_raster = np.where(np.isnan(out_raster), nodata_val, out_raster)
        
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=nodata_val)
        
        with rasterio.open(out_path, "w", **profile) as dest:
            dest.write(out_raster.astype("float32"), 1)


out_path = os.path.join(modeled, "HMS_TI_inches.tif")
if os.path.exists(out_path):
    print(f"Skipping existing raster: {out_path}")
else:
    with rasterio.open(HMS_TI) as src:
        raster_data = src.read(1, masked=True).filled(np.nan)
        out_raster = raster_data * 0.0254
        
        # Replace NaN with NoData value
        nodata_val = -9999
        out_raster = np.where(np.isnan(out_raster), nodata_val, out_raster)
        
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=nodata_val)
        
        with rasterio.open(out_path, "w", **profile) as dest:
            dest.write(out_raster.astype("float32"), 1)



rasters = {
    "HMS_EB": glob.glob(os.path.join(modeled, "*EB_inches*.tif")),
    "HMS_TI": glob.glob(os.path.join(modeled, "*TI_inches*.tif")),
    "iSnobal": glob.glob(os.path.join(modeled, "*mores_creek*.tif")),
    "SnowModel": glob.glob(os.path.join(modeled, "*snod*.tif")),
}

print(rasters)


MCS = os.path.join(dir, "MCS_outline/basin_outline.shp")

with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


stats_list = []

# Loop by model
for model, raster_list in rasters.items():
    out_name = f"{model}_basin_clip.tif"
    out_path = os.path.join(rasters_dir, out_name)

    if os.path.exists(out_path):
        print(f"Skipping existing LiDAR raster: {out_name}")
    else:
        for raster in raster_list:
            with rasterio.open(raster) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
# Copy the old profile and update it with new metadata
                profile = src.profile
                profile.update({
                 "driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform, # <-- Use the new transform from the mask operation
                 "nodata": -9999 # Explicitly set nodata if not already
                 })
            with rasterio.open(out_path, "w", **profile) as dest:
                    dest.write(out_image)

                

# Compute statistics
        data = out_image  # 1 band raster, extract 2D array
        mask = (data == -9999)
        data_masked = np.ma.array(data, mask=mask)  # mask nodata

        raster_stats = {
            "file": out_name,
            "model": model,
            "min": data_masked.min(),
            "mean": data_masked.mean(),
            "max": data_masked.max(),
            "zeros": np.sum(data_masked == 0)
        }
        
        stats_list.append(raster_stats)


# Convert stats to a DataFrame
stats_df = pd.DataFrame(stats_list)
stats_csv = os.path.join(figs_dir, "basin_stats.csv")
stats_df.to_csv(stats_csv, index=False)



rasters = {
    "HMS Energy Balance": glob.glob(os.path.join(rasters_dir, "*HMS_EB_basin_clip*.tif")),
    "HMS Temperature Index": glob.glob(os.path.join(rasters_dir, "*HMS_TI_basin_clip*.tif")),
    "iSnobal": glob.glob(os.path.join(rasters_dir, "*iSnobal_basin_clip*.tif")),
    "SnowModel": glob.glob(os.path.join(rasters_dir, "*SnowModel_basin_clip*.tif")),
}


#create figure of basin results

lidar_shp = gpd.read_file(os.path.join(dir, "MCS_outline/MCS_outline.shp"))

fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharex=True, sharey=True, constrained_layout=True)
axes = axes.flatten()  # flatten to 1D array for easy looping

for i, (model, raster_list) in enumerate(rasters.items()):
    raster_path = raster_list[0] 
    
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)  # read first band, mask NoData
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    im = axes[i].imshow(data, cmap="viridis",
                        extent=extent,
                        vmin=0,vmax=4.0 )
    axes[i].set_title(model)
    axes[i].grid(False)
    lidar_shp.plot(ax=axes[i], facecolor='none', edgecolor='black', linewidth=1.5)


# Add colorbar

# --- 5. Add a single, shared colorbar ---

# Add axes for the colorbar [left, bottom, width, height]
cbar_ax = fig.add_axes([0.92, 0.15, 0.025, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Snow Depth (m)', fontsize=14)

fig.suptitle(f"Mores Creek Basin Snow Depth, {date_obj.strftime('%B %d, %Y')}", fontsize=16)

fig.subplots_adjust(
    wspace=0.05,
    hspace=0.06,
    right=0.88

)

plt.savefig(os.path.join(figs_dir, "Basin_models.png"), dpi=300, bbox_inches="tight")

plt.show()

dfs = []

for model, raster_list in rasters.items():
    for raster in raster_list:
        with rasterio.open(raster) as src:
            data = src.read(1, masked=True)
            mask = (data == -9999)
            data_masked = np.ma.array(data, mask=mask)  # mask nodata
            flattened = data_masked.compressed()
            
        # convert to DataFrame
            df = pd.DataFrame({
            "Model": model,        # this column will store model names
            "value": flattened     # this column stores raster values
        })
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)




fig, ax = plt.subplots(figsize=(10, 6))

# Suppose your DataFrame has a column "Model" with long names
all_data.boxplot(column="value", by="Model", ax=ax, grid=False)

# Remove pandas' default title
ax.set_title("")

# Create multi-line labels by inserting '\n'
labels = [label.get_text() for label in ax.get_xticklabels()]
new_labels = []
for lbl in labels:
    if len(lbl) > 12:  # arbitrarily split long labels
        # split in half
        mid = len(lbl) // 2
        # find nearest space to split
        space_idx = lbl.rfind(" ", 0, mid)
        if space_idx == -1:
            space_idx = mid
        lbl = lbl[:space_idx] + "\n" + lbl[space_idx:].strip()
    new_labels.append(lbl)

ax.set_xticklabels(new_labels)

# Figure-level title
fig.suptitle(f"Mores Creek Snow Depth, {date_obj.strftime('%B %d, %Y')}", fontsize=16, y=0.95)

ax.set_ylabel("Snow Depth (m)")
ax.set_xlabel("")

#plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(figs_dir, "basin_boxplot.png"), dpi=300, bbox_inches="tight")
plt.show()



MCS = os.path.join(dir, "MCS_outline/MCS_outline.shp")

with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

rasters = {
    "HMS_EB": glob.glob(os.path.join(modeled, "*EB_inches*.tif")),
    "HMS_TI": glob.glob(os.path.join(modeled, "*TI_inches*.tif")),
    "iSnobal": glob.glob(os.path.join(modeled, "*mores_creek*.tif")),
    "SnowModel": glob.glob(os.path.join(modeled, "*snod*.tif")),
    "LiDAR": [lidar_raster],
    "LiDAR_2000": [lidar_raster_2000]
}



stats_list = []

# Loop by model
for model, raster_list in rasters.items():
    for raster in raster_list:
        out_name = f"{model}_MCS_clip.tif"
        out_path = os.path.join(rasters_dir, out_name)
        
        if os.path.exists(out_path):
            print(f"skipping creation of out_name")
            
        else:
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
            out_name = f"{model}_MCS_clip.tif"
            out_path = os.path.join(rasters_dir, out_name)

            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)

    # Compute statistics
            data = out_image[0]  # 1 band raster, extract 2D array
            mask = (data == -9999)
            data_masked = np.ma.array(data, mask=mask)  # mask nodata

            raster_stats = {
                "file": out_name,
                "model": model,
                "min": data_masked.min(),
                "mean": data_masked.mean(),
                "max": data_masked.max(),
                "zeros": np.sum(data_masked == 0)
            }
            
            stats_list.append(raster_stats)


# Convert stats to a DataFrame
stats_df = pd.DataFrame(stats_list)
stats_csv = os.path.join(figs_dir, "MCS_stats.csv")
stats_df.to_csv(stats_csv, index=False)



rasters = {
    "HMS Energy Balance": os.path.join(rasters_dir, "HMS_EB_MCS_clip.tif"),
    "HMS Temperature Index": os.path.join(rasters_dir, "HMS_TI_MCS_clip.tif"),
    "iSnobal": os.path.join(rasters_dir, "iSnobal_MCS_clip.tif"),
    "SnowModel": os.path.join(rasters_dir, "SnowModel_MCS_clip.tif"),
    "LiDAR": lidar_raster,
    "LiDAR_2000": lidar_raster_2000
}


fig, axes = plt.subplots(2, 3, figsize=(8, 10),constrained_layout=True)
axes = axes.flatten()  # flatten to 1D array for easy looping


axes = axes.flatten()

for i, (model, raster_path) in enumerate(rasters.items()):
    #raster_path = raster_list[0] 
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)  # read first band, mask NoData
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    im = axes[i].imshow(data, cmap="viridis", extent=extent, vmin=0,vmax=4.0) #aspect="auto")
    axes[i].set_title(model)
    axes[i].axis("off")

axes[-1].axis("off")
# Add colorbar
fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, label="Snow Depth")
fig.suptitle(f"Mores Creek Summit Snow Depth, {date_obj.strftime('%B %d, %Y')}", fontsize=14,  y=0.95)

# fig.subplots_adjust(
#     #wspace=.01,
#     hspace=0.01,
#     right=0.85
#     #top=0.90
# )

plt.savefig(os.path.join(figs_dir, "MCS_models.png"), dpi=300, bbox_inches="tight")

plt.show()


dfs = []

for model, raster in rasters.items():
    #for raster in raster_list:
        with rasterio.open(raster) as src:
            data = src.read(1, masked=True)
            mask = (data == -9999)
            data_masked = np.ma.array(data, mask=mask)  # mask nodata
            flattened = data_masked.compressed()
            
        # convert to DataFrame
            df = pd.DataFrame({
            "Model": model,        # this column will store model names
            "value": flattened     # this column stores raster values
        })
        dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)



fig, ax = plt.subplots(figsize=(10, 6))

# Suppose your DataFrame has a column "Model" with long names
all_data.boxplot(column="value", by="Model", ax=ax, grid=False)

# Remove pandas' default title
ax.set_title("")

# Create multi-line labels by inserting '\n'
labels = [label.get_text() for label in ax.get_xticklabels()]
new_labels = []
for lbl in labels:
    if len(lbl) > 12:  # arbitrarily split long labels
        # split in half
        mid = len(lbl) // 2
        # find nearest space to split
        space_idx = lbl.rfind(" ", 0, mid)
        if space_idx == -1:
            space_idx = mid
        lbl = lbl[:space_idx] + "\n" + lbl[space_idx:].strip()
    new_labels.append(lbl)

ax.set_xticklabels(new_labels)

# Figure-level title
fig.suptitle(f"Mores Creek Summit Snow Depth, {date_obj.strftime('%B %d, %Y')}", fontsize=16, y=0.95)

ax.set_ylabel("Snow Depth (m)")
ax.set_xlabel("")

#plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(figs_dir, "MCS_boxplot.png"), dpi=300, bbox_inches="tight")
plt.show()


##resample and difference models with LiDAR
## 100-m is lidar_raster, 2-km raster is lidar_raster_2000
##need to remove lidar form rasters dictionary

rasters = {
    "HMS Energy Balance": "HMS_EB_MCS_clip.tif",
    "HMS Temperature Index": "HMS_TI_MCS_clip.tif",
    "iSnobal": "iSnobal_MCS_clip.tif",
    "SnowModel": "SnowModel_MCS_clip.tif",
}

lidar = os.path.join(rasters_dir, "LiDAR_MCS_clip.tif")
lidar_2000 = os.path.join(rasters_dir, "LiDAR_2000_MCS_clip.tif")

for model, filename in rasters.items():

    lidar_path = lidar_2000 if "HMS" in model else lidar
    model_path = os.path.join(rasters_dir, filename)

    with rasterio.open(lidar_path) as lidar_src, rasterio.open(model_path) as model_src:

        lidar_data = lidar_src.read(1, masked=True)

        reprojected = np.full(
            lidar_data.shape,
            -9999,
            dtype=np.float32
        )

        reproject(
            source=model_src.read(1),
            destination=reprojected,
            src_transform=model_src.transform,
            src_crs=model_src.crs or lidar_src.crs,
            dst_transform=lidar_src.transform,
            dst_crs=lidar_src.crs,
            src_nodata=-9999,
            dst_nodata=-9999,
            resampling=Resampling.nearest
        )

        mask = (
            np.ma.getmaskarray(lidar_data)
            | (reprojected == -9999)
        )

        model_resampled = np.ma.array(reprojected, mask=mask)
        diff = model_resampled - lidar_data

        profile = lidar_src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=-9999,
            compress="lzw"
        )

    model_tag = model.replace(" ", "_")

    with rasterio.open(
        os.path.join(rasters_dir, f"{model_tag}_lidar_resample.tif"),
        "w",
        **profile
    ) as dst:
        dst.write(model_resampled.filled(-9999).astype(np.float32), 1)

    with rasterio.open(
        os.path.join(rasters_dir, f"{model_tag}_lidar_diff.tif"),
        "w",
        **profile
    ) as dst:
        dst.write(diff.filled(-9999).astype(np.float32), 1)





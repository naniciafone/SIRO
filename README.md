# Snow Informed Reservoir Operations Model Intercomparison Experiment

### Overview
This repository will house scripts for data preparation and analysis. 
Descriptions of where to find what will be added soon


## Table of Contents
- [LiDAR Prepration]()
- [Model Comparison]()

## LiDAR Prep

## Model Comparison

### Directory Structure

Directories should be organized as follows, with folders organized by data:
   ```
   Date/
   └── lidar/
       ├── "lidar file"/
   └── modeled/
       ├── HMS Energy Balance
       └── HMS
       └── SnowModel
       └── iSnobal
   └── outputs/
   └── MCS_outline

   ```

### Assign Directory Path and call modeled and LiDAR data

```bash
dir = "your directory here"
lidar = glob.glob(os.path.join(dir, "lidar","*.tif"))
modeled = os.path.join(dir, "modeled")

rasters = (
    glob.glob(os.path.join(modeled, "*EB_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*TI_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*thickness*.tif")) +
    glob.glob(os.path.join(modeled, "*snod*.tif"))
)

rasters = (
    glob.glob(os.path.join(modeled, "*EB_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*TI_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*thickness*.tif")) +
    glob.glob(os.path.join(modeled, "*snod*.tif"))
)
```

### Look at Basin Wide Data

Call Mores Creek Summit Shapefile

```bash
MCS = os.path.join(dir, "MCS_outline/basin_outline.shp")
```

And mask modeled inputs by MCS shapefile. Masked inputs are saved to "outputs" directory:

```bash
with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

for raster in rasters:
    with rasterio.open(raster) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    
    out_dir = os.path.join(dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.basename(raster).replace(".tif", "_MCS.tif")
    out_path = os.path.join(out_dir, out_name)

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)
```

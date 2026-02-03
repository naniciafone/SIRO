# Snow Informed Reservoir Operations Model Intercomparison Experiment

### Overview
This repository will house scripts for data preparation and analysis. 
Descriptions of where to find what will be added soon


## Table of Contents
- [Data Prep]()
- [Model Comparison]()

## Data Prep

### LiDAR

Raw LiDAR geotiffs were resampled from 0.5-m to 100-m using bilinear, cubic, and average resampling. No significant difference
between the average and spread of resampled outputs was found. However, average resampling may populate a pixel with sparse
data and therefore misrepresent the dataset. As areas of sparse data often correlate with noise, it is better to classify
sparse pixels as NA. As bilinear and cubic resampling both retain NA values when downsampling, these methods were chosen over average 
resampling. 

LiDAR data was filtered and then iteratively downsampled using cubic resampling with ```GDAL.warp   ```. All values 
less than *-0.2 meters* and greater than *10 meters* were set to NAN. The lower threshold was selected after investigating 
along Highway 21, the corridor used for co-registration. All LiDAR rasters were clipped to the co-registration region 
using the same clipping geometry as is used in ice-road-copters, and the average, min, max, and SD of clipped rasters were 
extracted to understand the variability in offsets along the "no change"  region. For all clipped rasters, the mean minus 
one SD was greater than *-0.2 meters*. Once rasters were filtered by value,  rasters were then iteratively resampled from 
0.5-meter -> 1-meter -> 10-meter -> 100-meter. Remaining negative pixels were set to 0. 

## Model Comparison

### Directory Structure

Directories should be organized as follows, with folders organized by data:
   ```
   Date/
   └── lidar/
       ├── "lidar file"/
   └── modeled/
       ├── HMS Energy Balance geotiff
       └── HMS Temperature Index geotiff
       └── SnowModel geotiff
       └── iSnobal geotiff
   └── outputs
   └── MCS_outline
          └── MCS_outline.shp
          └── basin_outline.ship

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

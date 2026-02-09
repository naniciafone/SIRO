# Snow Informed Reservoir Operations Model Intercomparison Experiment

### Overview
This repository will house scripts for data preparation and analysis. 
Descriptions of where to find what will be added soon


## Table of Contents
- Data Prep
- Model Comparison

## Data Prep

### LiDAR

Raw LiDAR geotiffs were resampled from 0.5-m to 100-m using bilinear, cubic, and average resampling. No significant difference
between the average and spread of resampled outputs was found. However, average resampling may populate a pixel with sparse
data and therefore misrepresent the dataset. As areas of sparse data often correlate with noise, it is better to classify
sparse pixels as NA. As bilinear and cubic resampling both retain NA values when downsampling, these methods were chosen over average 
resampling. 

LiDAR data was filtered and then iteratively downsampled using cubic resampling with ```GDAL.warp```. All values 
less than *-0.2 meters* and greater than *10 meters* were set to NAN. The lower threshold was selected after investigating 
along Highway 21, the corridor used for co-registration. All LiDAR rasters were clipped to the co-registration region 
using the same clipping geometry as is used in ice-road-copters, and the average, min, max, and SD of clipped rasters were 
extracted to understand the variability in offsets along the "no change"  region. For all clipped rasters, the mean minus 
one SD was greater than *-0.2 meters*. Once rasters were filtered by value,  rasters were then iteratively resampled from 
0.5-meter -> 1-meter -> 10-meter -> 100-meter. Remaining negative pixels were set to 0. 

### HMS Resampling

Per a conversation with the HMS team, HMS was run at 2000-m and 100-m resolution with no appreciable difference in computed results.
Both TI and ET implementations of HMS are "heavily dependent on the resolution of the met data used as boundary condition". HRRR 
data was already interpolated from 3-km to 2-km, adding "an artificial increase in precision but not accuracy". The HMS team therefore
elected to run the model at 2,000-m resolution rather than 100-m, and I have resampled the data for model comparison.
The resampling method can be found in [scripts/resample.py](scripts/resample.py).


## Model Preparation

The script used to difference modeled outputs and LiDAR data can be found at [scripts/Model_prep.py](scripts/Model_prep.py). The following subsections describe a breakdown of this script.


### Directory Structure

Directories should be organized as follows, with folders organized by date. ```MCS_outline.shp``` is the LiDAR domain and ```basin_outline.shp``` is teh Mores Creek Summit Basin.
   ```
   Date/
   └── lidar/
       ├── "lidar file"
   └── modeled/
       └── Task1/
           └── "HMS Energy Balance geotiff"
           └── "HMS Temperature Index geotiff"
           └── "SnowModel geotiff"
           └── "iSnobal geotiff"
       └── Task2/
           └── "HMS Energy Balance geotiff"
           └── "HMS Temperature Index geotiff"
           └── "SnowModel geotiff"
           └── "iSnobal geotiff"
       └── SNODAS
            └── basin_clip
            └── MCS_clip
   └── outputs/
       └── task1/
            └── rasters/
            └── figs/
       └── task2/
            └── rasters/
            └── figs/
   └── MCS_outline/
       └── MCS_outline.shp
       └── basin_outline.shp
   └── Model_prep.py

   ```
### Script Breakdown
[Model_prep.py](scripts/Model_prep.py) is set to run in the folder in which it is located. ```Task_numer``` must be set to 1 or 2.  

<details>
  <summary>Directory set up</summary>

dir = "."

Task_number = 1
</details>

All LiDAR NANs are set to ```-9999```, and the date is extracted from the LiDAR file path.

<details>
  <summary>LiDAR Prep </summary>

Set up directory structure and extract date:

```
dir = "."
lidar = glob.glob(os.path.join(dir, "lidar","*.tif"))

parts = os.path.basename(lidar[0]).split("_")
for part in parts:
    if part.isdigit() and len(part) == 8:
        date_str = part
        break

date_obj = datetime.strptime(date_str, "%Y%m%d")

f os.path.exists(outfile):
    print(f"Skipping existing LiDAR raster: {outfile}")
    lidar_raster = outfile
else:
    with rasterio.open(lidar[0]) as src:
        data = src.read(1, masked=True)
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=-9999)
        outfile = os.path.join(dir, "lidar", "lidar_"+date_str+".tif")

 ```
Replace NANs with  ```-9999 ```

 ```
data_filled = np.where(np.isnan(data), -9999, data)
with rasterio.open(outfile, "w", **profile) as dst:
    dst.write(data_filled.astype("float32"), 1)  
lidar_raster = outfile
 ```

</details>



HMS outputs must be converted from inches to centimeters, and again, all NANs are set to ```-9999 ```. These geotiffs are 
written to the ```modeled/``` directory. 

<details>
  <summary> HMS Prep </summary>

```
if Task_number == 1:
    modeled = os.path.join(dir, "modeled/Task1")
else:
    modeled = os.path.join(dir, "modeled/Task2")

HMS_EB = glob.glob(os.path.join(modeled, "*EB_snow_depth*.tif"))[0]
HMS_TI = glob.glob(os.path.join(modeled, "*TI_snow_depth*.tif"))[0]

with rasterio.open(HMS_EB) as src:
    raster_data = src.read(1, masked=True).filled(np.nan)
    out_raster = raster_data * 0.0254
    
    # Replace NaN with NoData value
    nodata_val = -9999
    out_raster = np.where(np.isnan(out_raster), nodata_val, out_raster)
    
    out_path = os.path.join(modeled, "HMS_EB_inches.tif")
    profile = src.profile
    profile.update(dtype=rasterio.float32, nodata=nodata_val)
    
    with rasterio.open(out_path, "w", **profile) as dest:
        dest.write(out_raster.astype("float32"), 1)

with rasterio.open(HMS_TI) as src:
    raster_data = src.read(1, masked=True).filled(np.nan)
    out_raster = raster_data * 0.0254
    
    # Replace NaN with NoData value
    nodata_val = -9999
    out_raster = np.where(np.isnan(out_raster), nodata_val, out_raster)
    
    out_path = os.path.join(modeled, "HMS_TI_inches.tif")
    profile = src.profile
    profile.update(dtype=rasterio.float32, nodata=nodata_val)
    
    with rasterio.open(out_path, "w", **profile) as dest:
        dest.write(out_raster.astype("float32"), 1)
```

</details>

Rasters for all models are called from the ```modeled/``` folder using ```glob``` to query the unique filename. The
Mores Creek Summit Basin shapefile is called from ```MCS_outline/```. Modeled inputs are clipped to the basin shapefile 
using the ```fiona``` package, and clipped outputs are stored in ```outputs/```. For each clipped raster,
the raster minimum, maximum, mean, and total number of 0 pixels (snow free pixels) are written to a dataframe that is stored
as a csv in ```outputs/```. 



<details>
  <summary> Basin-Wide Analysis </summary>

```
rasters = {
    "HMS_EB": glob.glob(os.path.join(modeled, "*EB_inches*.tif")),
    "HMS_TI": glob.glob(os.path.join(modeled, "*TI_inches*.tif")),
    "iSnobal": glob.glob(os.path.join(modeled, "*thickness*.tif")),
    "SnowModel": glob.glob(os.path.join(modeled, "*snod*.tif")),
}

# Call shapefile
MCS = os.path.join(dir, "MCS_outline/basin_outline.shp")
with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

# Set up directory structure
if Task_number == 1:
    out_dir = os.path.join(dir, "outputs/task1/")
else:
    out_dir = os.path.join(dir, "outputs/task2/")

rasters_dir = os.path.join(out_dir, "rasters/")
figs_dir = os.path.join(out_dir, "figs/")

out_dir = os.path.join(dir, "outputs")
stats_list = []

#Loop by model

for model, raster_list in rasters.items():
    for raster in raster_list:
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            profile = src.profile

        # Include model in output filename
        out_name = f"{model}_basin_clip.tif"
        out_path = os.path.join(out_dir, out_name)

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
stats_csv = os.path.join(out_dir, "basin_stats.csv")
stats_df.to_csv(stats_csv, index=False)

```
</details>

A figure of basin-wide rasters and a boxplot of basin-wide values are generated and stored in ```outputs/``` (not included in code block below).
Modeled outputs and LiDAR data are all clipped to the LiDAR domain. Clipped rasters are stored in ```outputs/```. Similarly, a figure 
of clipped rasters and a boxplot of clipped raster values is stored in the same directory, but this code is not included below. 

<details>
  <summary> LiDAR-Domain Analysis </summary>

```
MCS = os.path.join(dir, "MCS_outline/MCS_outline.shp")

with fiona.open(MCS, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

rasters = {
    "HMS_EB": glob.glob(os.path.join(modeled, "*EB_inches*.tif")),
    "HMS_TI": glob.glob(os.path.join(modeled, "*TI_inches*.tif")),
    "iSnobal": glob.glob(os.path.join(modeled, "*thickness*.tif")),
    "SnowModel": glob.glob(os.path.join(modeled, "*snod*.tif")),
    "LiDAR": [lidar_raster]
}

for model, raster_list in rasters.items():
    for raster in raster_list:
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

```

</details>

Resample all model rasters to match LiDAR grid, and subtract model outputs from LiDAR data.

<details>
  <summary> LiDAR-Domain Analysis </summary>

```
del rasters["LiDAR"]
lidar = os.path.join(rasters_dir, "LiDAR_MCS_clip.tif")


with rasterio.open(lidar) as src:
    lidar_data = src.read(1, masked=True)
    profile = src.profile                 # for writing outputs
    lidar_crs = src.crs
    lidar_transform = src.transform 

for model, raster_path in rasters.items():
    with rasterio.open(raster_path) as src:
        model_data = src.read(1, masked=True)
        model_transform = src.transform
        model_crs = src.crs

        # Assign CRS if missing
        if model_crs is None:
            model_crs = lidar_crs
            print(f"Assigned CRS {model_crs} to {model} because it was missing.")

        # Prepare array for reprojected data
        reprojected_model = np.empty(lidar_data.shape, dtype=np.float32)

        # Reproject/resample model to match LiDAR
        reproject(
            source=model_data,
            destination=reprojected_model,
            src_transform=model_transform,
            src_crs=model_crs,
            dst_transform=lidar_transform,
            dst_crs=lidar_crs,
            resampling=Resampling.bilinear
        )

        # Mask wherever either raster is NaN / masked
        combined_mask = (lidar_data.data == -9999) | (reprojected_model == -9999)
        model_masked = np.ma.array(reprojected_model, mask=combined_mask)
        lidar_masked = np.ma.array(lidar_data, mask=combined_mask)

        # Compute difference
        diff_data = lidar_masked - model_masked

        # Write difference raster
        out_profile = profile.copy()
        out_profile.update(dtype=rasterio.float32, compress="lzw")

        # Write difference raster
        out_path = os.path.join(rasters_dir, f"{model.replace(' ', '_')}_lidar_diff.tif")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(diff_data.filled(np.nan).astype(np.float32), 1)

```

</details>
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

### Assign Directory Path and Call Data
```bash
dir = "your directory here"

```bash
lidar = glob.glob(os.path.join(dir, "lidar","*.tif"))
modeled = os.path.join(dir, "modeled")

rasters = (
    glob.glob(os.path.join(modeled, "*EB_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*TI_snow_depth*.tif")) +
    glob.glob(os.path.join(modeled, "*thickness*.tif")) +
    glob.glob(os.path.join(modeled, "*snod*.tif"))
)
```

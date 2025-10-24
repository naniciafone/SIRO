from pathlib import Path
import geopandas as gpd
import subprocess

basin = gpd.read_file("Path to shapefile")

## build 7k buffer around basin
buffer_distance = 7000
basin_buffered = basin.copy() # Create a copy to avoid modifying original
basin_buffered['geometry'] = basin_buffered['geometry'].buffer(buffer_distance)

##Find rectangular polygon around buffer
xmin,ymin,xmax,ymax =  basin_buffered.total_bounds


##Prep rasters
home_dir = Path(__file__).parent
out_dir = Path(__file__).parent / "reproject"
out_dir.mkdir(parents=True, exist_ok=True)

##EVT
EVT_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-tr", "100", "100",
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "LC22_EVT_230.tif"),
    str(out_dir / "LC22_EVT_230.tif")]

print(" ".join(EVT_cmd))

subprocess.run(EVT_cmd, check=True)

##DEM
DEM_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-tr", "100", "100",
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "merged.tif"),
    str(out_dir / "DEM.tif")]

print(" ".join(DEM_cmd))

subprocess.run(DEM_cmd, check=True)

##FH
FH_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-tr", "100", "100",
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "Forest_height_2019_NAM.tif"),
    str(out_dir / "Forest_height_2019_NAM.tif")]

print(" ".join(FH_cmd))
subprocess.run(FH_cmd, check=True)


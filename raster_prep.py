from pathlib import Path
import geopandas as gpd

basin = gpd.read_file("MCS_outline/basin_outline.shp")

## build 7k buffer around basin
buffer_distance = 7000
basin_buffered = basin.copy() # Create a copy to avoid modifying original
basin_buffered['geometry'] = basin_buffered['geometry'].buffer(buffer_distance)

##Find rectangular polygon around buffer
xmin,ymin,xmax,ymax =  basin_buffered.total_bounds


##Prep rasters
home_dir = Path(__file__).parent
out_dir = home_dir / "GIS" / "reproject"
out_dir.mkdir(parents=True, exist_ok=True)

##EVT
EVT_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "LF2022_EVT_230_CONUS/LF2022_EVT_230_CONUS/Tif/LC22_EVT_230.tif"),
    str(out_dir / "LC22_EVT_230.tif")]

print(" ".join(EVT_cmd))

##DEM
DEM_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "GIS/merged.tif"),
    str(out_dir / "DEM.tif")]

print(" ".join(DEM_cmd))

##FH
FH_cmd = [
    "gdalwarp",
    "-t_srs", "EPSG:32611",
    "-te", str(xmin), str(ymin), str(xmax), str(ymax),
    "-r", "near",
    "-co", "COMPRESS=LZW",
    "-co", "TILED=YES",
    "-co", "BIGTIFF=YES",   # handle large files
    str(home_dir / "GIS/Forest_height_2019_NAM.tif"),
    str(out_dir / "Forest_height_2019_NAM.tif")]

print(" ".join(FH_cmd))



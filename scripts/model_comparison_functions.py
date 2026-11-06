import os
#os.environ['PROJ_DATA'] = "C:/Users/RDCRLSMC/AppData/Local/miniconda3/envs/SIRO/Library/share/proj"
import re
import glob
from osgeo import gdal
from pyproj import Proj, Transformer

# Output file tags written by Model_prep.py -> dataset names used in figures and tables
MODEL_NAMES = {"HMS_EB": "HMS-EB", "HMS_TI": "HMS-TI", "iSnobal": "iSnobal", "SnowModel": "SnowModel"}
LIDAR_NAMES = {"LiDAR": "LiDAR", "LiDAR_2000": "LiDAR-2000"}


def x_y_snotel(snotel_station):
    """
    Convert Snotel from lon/lat to x/y
    """
    converter = Transformer.from_proj(Proj('EPSG:4326'), Proj('EPSG:32611'), always_xy=True)
    return converter.transform(snotel_station.metadata.x, snotel_station.metadata.y)

def find_date_directories(start_dir):
    """
    Finds all directories within start_dir whose names are exactly 6 digits.

    Args:
        start_dir (str): The path to the directory to start searching from.

    Returns:
        list: A list of full paths to the matching directories.
    """
    # Regex pattern to match exactly 6 digits
    # ^ asserts the start of the string, \\d{6} matches exactly six digits,
    # and $ asserts the end of the string.
    pattern = re.compile(r"^\d{8}$")
    matching_directories = []

    # os.walk traverses the directory tree recursively
    for root, dirs, files in os.walk(start_dir):
        # We only need to check the directory names (dirs) in the current root
        for dir_name in dirs:
            # Check if the directory name matches the 6-digit pattern
            if pattern.match(dir_name):
                # If it matches, construct the full path and add to the list
                full_path = os.path.join(root, dir_name)
                matching_directories.append(full_path)

    return matching_directories


def process_all_dates(parent_dir, task_number):
    """
    Processes multiple date-named directories to build a nested dictionary of model outputs.

    Args:
        parent_dir (str): The parent directory containing subdirectories named 'YYYYMMDD'.
        task_number (int): The task number (1 or 2).

    Returns:
        dict: A nested dictionary where keys are dates and values are dictionaries
              of model rasters for that date.
    """
    date_dirs = find_date_directories(parent_dir)
    modeled_data = {}

    for dir in date_dirs:
        date_str = os.path.basename(dir)
        modeled_data[date_str] = {}

        raster = glob.glob(os.path.join(dir, 'modeled/SNODAS', "*basin_clip.tif"))
        if raster:
            modeled_data[date_str]['SNODAS-basin'] = raster[0]

        raster = glob.glob(os.path.join(dir, 'modeled/SNODAS', '*MCS_clip.tif'))
        if raster:
            modeled_data[date_str]['SNODAS-MCS'] = raster[0]

        if task_number == 1:
            output_dir = os.path.join(dir, "outputs/task1/rasters")
        else:
            output_dir = os.path.join(dir, "outputs/task2/rasters")

        # create lidar dictionary
        raster = glob.glob(os.path.join(output_dir, "*LiDAR_MCS_clip*.tif"))
        if raster:
            modeled_data[date_str]['lidar'] = raster[0]

        # create lidar 2000-m dictionary
        raster = glob.glob(os.path.join(output_dir, "*LiDAR_2000_MCS_clip*.tif"))
        if raster:
            modeled_data[date_str]['lidar_2000'] = raster[0]

        # create dcitionary of outputs clipped to basin
        models_basin = {}
        raster_search_pattern = os.path.join(output_dir, "*_basin_clip.tif")
        for raster_path in glob.glob(raster_search_pattern):
            file_name = os.path.basename(raster_path)
            model_name = file_name.replace("_basin_clip.tif", "").replace("_", "-")
            models_basin[model_name] = raster_path

        if models_basin:
            modeled_data[date_str]['basin_clip'] = models_basin

            # create dictionary of outputs clipped to lidar domain
        models_lidar = {}
        raster_search_pattern = os.path.join(output_dir, "*_MCS_clip.tif")
        for raster_path in glob.glob(raster_search_pattern):
            file_name = os.path.basename(raster_path)
            model_name = file_name.replace("_MCS_clip.tif", "").replace("_", "-")
            models_lidar[model_name] = raster_path

        if models_lidar:
            modeled_data[date_str]['lidar_clip'] = models_lidar

            # create dictionary of outputs clipped to lidar domain and resampled
        models_resample = {}
        raster_search_pattern = os.path.join(output_dir, "*lidar_resample.tif")
        for raster_path in glob.glob(raster_search_pattern):
            file_name = os.path.basename(raster_path)
            model_name = file_name.replace("_lidar_resample.tif", "").replace("_", "-")
            models_resample[model_name] = raster_path

        if models_resample:
            modeled_data[date_str]['resample'] = models_resample

        # create dictionary of outputs differenced from lidar data
        models_diff = {}
        raster_search_pattern = os.path.join(output_dir, "*_lidar_diff.tif")
        for raster_path in glob.glob(raster_search_pattern):
            file_name = os.path.basename(raster_path)
            model_name = file_name.replace("_lidar_diff.tif", "").replace("_", "-")
            models_diff[model_name] = raster_path
        if models_diff:
            modeled_data[date_str]['diff'] = models_diff

    return modeled_data


def get_model_prep_outputs(parent_dir, task_number):
    """
    Collect the rasters written by Model_prep.py for one task, by exact filename.

    Unlike process_all_dates(), this looks up each expected file by name rather than by
    wildcard, so leftover files from older runs (e.g. HMS_Energy_Balance_lidar_resample.tif)
    are never picked up. Missing files are reported and left out.

    Args:
        parent_dir (str): The parent directory containing subdirectories named 'YYYYMMDD'.
        task_number (int): The task number (1 or 2).

    Returns:
        dict: {date: {"lidar": path, "lidar_2000": path, "meters": {name: path} (HMS only),
                      "basin_clip": {name: path}, "lidar_clip": {name: path},
                      "resample": {name: path}, "diff": {name: path}}}
              where names are the values of MODEL_NAMES (and LIDAR_NAMES for "lidar_clip").
    """
    outputs = {}
    for date_dir in sorted(find_date_directories(parent_dir)):
        date_str = os.path.basename(date_dir)
        rasters_dir = os.path.join(date_dir, "outputs", f"task{task_number}", "rasters")
        expected = {
            "lidar": os.path.join(rasters_dir, "LiDAR_MCS_clip.tif"),
            "lidar_2000": os.path.join(rasters_dir, "LiDAR_2000_MCS_clip.tif"),
            "meters": {name: os.path.join(rasters_dir, f"{tag}_meters.tif") for tag, name in MODEL_NAMES.items()
                       if tag.startswith("HMS")},
            "basin_clip": {name: os.path.join(rasters_dir, f"{tag}_basin_clip.tif") for tag, name in MODEL_NAMES.items()},
            "lidar_clip": {name: os.path.join(rasters_dir, f"{tag}_MCS_clip.tif")
                           for tag, name in {**LIDAR_NAMES, **MODEL_NAMES}.items()},
            "resample": {name: os.path.join(rasters_dir, f"{tag}_lidar_resample.tif") for tag, name in MODEL_NAMES.items()},
            "diff": {name: os.path.join(rasters_dir, f"{tag}_lidar_diff.tif") for tag, name in MODEL_NAMES.items()},
        }

        found, missing = {}, []
        for key, value in expected.items():
            if isinstance(value, dict):
                found[key] = {name: path for name, path in value.items() if os.path.exists(path)}
                missing += [path for path in value.values() if not os.path.exists(path)]
            elif os.path.exists(value):
                found[key] = value
            else:
                missing.append(value)
        if missing:
            print(f"{date_str} task {task_number}: {len(missing)} expected Model_prep.py outputs missing, "
                  f"e.g. {os.path.basename(missing[0])}")
        outputs[date_str] = found

    return outputs


def get_raw_data(parent_dir, task_number):
    date_dirs = find_date_directories(parent_dir)
    raw_data = {}
    for dir in date_dirs:
        date_str = os.path.basename(dir)
        raw_data[date_str] = {}

        if task_number == 1:
            dir_rasters = os.path.join(dir, "modeled/Task1")
        else:
            dir_rasters = os.path.join(dir, "modeled/Task2")

        raster = glob.glob(os.path.join(dir_rasters, "*HMS_TI*.tif"))
        if raster:
            raw_data[date_str]['HMS_TI'] = raster[0]
        raster = glob.glob(os.path.join(dir_rasters, "*HMS_EB*.tif"))
        if raster:
            raw_data[date_str]['HMS_EB'] = raster[0]
        raster = glob.glob(os.path.join(dir_rasters, "*snod*.tif"))
        if raster:
            raw_data[date_str]['Snow Model'] = raster[0]

        raster = glob.glob(os.path.join(dir_rasters, "*mores_creek*.tif"))
        if raster:
            raw_data[date_str]['iSnobal'] = raster[0]

    return raw_data


def get_raster_pixel_value(file, geo_x, geo_y):
    """
    Return value from raster at given geo coordinates
    """
    dataset = gdal.Open(file)
    gt = dataset.GetGeoTransform()

    pixel_coords = gdal.ApplyGeoTransform(
        gdal.InvGeoTransform(gt), geo_x, geo_y
    )
    pixel_x, pixel_y = int(pixel_coords[0]), int(pixel_coords[1])

    band = dataset.GetRasterBand(1)
    val_array = band.ReadAsArray(
        xoff=pixel_x, yoff=pixel_y, win_xsize=1, win_ysize=1
    )

    return val_array[0][0]
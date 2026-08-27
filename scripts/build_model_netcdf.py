"""
Build per-water-year netCDFs (SD, SWE) from folders of dated GeoTIFFs.

Mirrors the SnowModel netCDF format in time_series/4_SM: one file per
water year per task, dims (time, y, x), data variables SD (m) and SWE
(mm), CRS hard-coded onto the output.

Every .tif/.tiff under the given "<model-name>_Task<N>" folder is
collected recursively regardless of subfolder naming; each file is
classified as SD or SWE by matching "depth"/"swe" in its filename, and
its date is pulled from the filename with --date-regex. Files are then
bucketed into water years (Oct 1 by default) purely by date, so odd
subfolder organization (extra dumps, inconsistent "depth" vs
"snow_depth" naming, etc.) doesn't need to be special-cased -- it just
needs to not exist if you want it excluded (delete it first).

Usage:
    python build_model_netcdf.py --model-dir "C:/.../1_TI" --model-name TI \
        --depth-factor 0.0254 --swe-factor 25.4 --dry-run

    python build_model_netcdf.py --model-dir "C:/.../3_iSnobal" --model-name iSnobal
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import rioxarray  # noqa: F401 -- registers the .rio accessor used below

DEFAULT_DATE_REGEX = r"(\d{4})[-_](\d{2})[-_](\d{2})T"


def find_task_dir(model_dir: Path, model_name: str, task: int):
    candidates = {f"{model_name}_task{task}".lower(), f"task{task}".lower()}
    for p in model_dir.iterdir():
        if p.is_dir() and p.name.lower() in candidates:
            return p
    return None


def classify_variable(filename: str):
    name = filename.lower()
    is_depth = "depth" in name
    is_swe = "swe" in name
    if is_depth and not is_swe:
        return "SD"
    if is_swe and not is_depth:
        return "SWE"
    return None


def water_year(date: datetime, start_month: int) -> int:
    return date.year + 1 if date.month >= start_month else date.year


def collect_files(task_dir: Path, date_regex: str, start_month: int):
    """Returns {water_year: {"SD": {date: path}, "SWE": {date: path}}}."""
    pattern = re.compile(date_regex)
    buckets = defaultdict(lambda: {"SD": {}, "SWE": {}})
    skipped = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        var = classify_variable(path.name)
        m = pattern.search(path.name)
        if var is None or not m:
            skipped.append(path)
            continue
        y, mo, d = (int(g) for g in m.groups())
        date = datetime(y, mo, d)
        wy = water_year(date, start_month)
        existing = buckets[wy][var].get(date)
        if existing is not None:
            raise ValueError(
                f"Duplicate date {date.date()} for {var} in WY{wy % 100:02d}: "
                f"'{existing}' and '{path}' both map to this date."
            )
        buckets[wy][var][date] = path
    if skipped:
        print(f"  WARNING: skipped {len(skipped)} file(s) with no recognizable variable/date:")
        for p in skipped[:10]:
            print(f"    {p}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")
    return buckets


def read_stack(dated_paths: dict, nodata_value: float, factor: float):
    dates = sorted(dated_paths)
    arrays = []
    profile = None
    for date in dates:
        with rasterio.open(dated_paths[date]) as src:
            if profile is None:
                profile = src.profile
            elif (src.width, src.height) != (profile["width"], profile["height"]):
                raise ValueError(
                    f"Grid shape mismatch at {dated_paths[date]}: "
                    f"expected {(profile['width'], profile['height'])}, got {(src.width, src.height)}"
                )
            data = src.read(1).astype("float32")
            src_nodata = src.nodata
            mask = np.isclose(data, src_nodata) if src_nodata is not None else np.zeros_like(data, dtype=bool)
            data = np.where(mask, nodata_value, data * factor).astype("float32")
        arrays.append(data)
    return dates, np.stack(arrays, axis=0), profile


def build_dataset(depth_paths, swe_paths, nodata_value, depth_factor, swe_factor, crs, pad_mismatched_dates):
    sd_all, swe_all = set(depth_paths), set(swe_paths)
    if pad_mismatched_dates:
        common_dates = sorted(sd_all | swe_all)
    else:
        common_dates = sorted(sd_all & swe_all)
        dropped = sorted(sd_all ^ swe_all)
        if dropped:
            print(f"  WARNING: {len(dropped)} date(s) present in only one of SD/SWE, dropped to keep the overlap: "
                  f"{[d.date() for d in dropped[:5]]}{' ...' if len(dropped) > 5 else ''}")
        depth_paths = {d: depth_paths[d] for d in common_dates}
        swe_paths = {d: swe_paths[d] for d in common_dates}

    sd_dates, sd_stack, profile = read_stack(depth_paths, nodata_value, depth_factor) if depth_paths else (None, None, None)
    swe_dates, swe_stack, swe_profile = read_stack(swe_paths, nodata_value, swe_factor) if swe_paths else (None, None, None)
    profile = profile or swe_profile
    if profile is None:
        raise ValueError("No SD or SWE files found for this water year.")

    if pad_mismatched_dates:
        missing_depth = sorted(set(common_dates) - set(sd_dates or []))
        missing_swe = sorted(set(common_dates) - set(swe_dates or []))
        if missing_depth:
            print(f"  WARNING: {len(missing_depth)} date(s) have SWE but no SD (filled with nodata): "
                  f"{[d.date() for d in missing_depth[:5]]}{' ...' if len(missing_depth) > 5 else ''}")
        if missing_swe:
            print(f"  WARNING: {len(missing_swe)} date(s) have SD but no SWE (filled with nodata): "
                  f"{[d.date() for d in missing_swe[:5]]}{' ...' if len(missing_swe) > 5 else ''}")

    transform = profile["transform"]
    width, height = profile["width"], profile["height"]
    x = transform.c + transform.a * (np.arange(width) + 0.5)
    y = transform.f + transform.e * (np.arange(height) + 0.5)

    def full_array(dates, stack):
        if dates is None:
            return np.full((len(common_dates), height, width), nodata_value, dtype="float32")
        if dates == common_dates:
            return stack
        out = np.full((len(common_dates), height, width), nodata_value, dtype="float32")
        idx = {d: i for i, d in enumerate(common_dates)}
        for i, d in enumerate(dates):
            out[idx[d]] = stack[i]
        return out

    # grid_mapping must be set explicitly on each data variable -- rioxarray's
    # write_crs() adds the spatial_ref coordinate but won't backfill this link,
    # and without it ds.rio.crs silently returns None on reopen even though the
    # CRS is technically present in the file.
    var_attrs = {"grid_mapping": "spatial_ref"}
    ds = xr.Dataset(
        {
            "SD": (("time", "y", "x"), full_array(sd_dates, sd_stack), {"units": "m", "long_name": "snow depth", **var_attrs}),
            "SWE": (("time", "y", "x"), full_array(swe_dates, swe_stack), {"units": "mm", "long_name": "snow water equivalent", **var_attrs}),
        },
        coords={"time": np.array(common_dates, dtype="datetime64[ns]"), "y": y, "x": x},
    )
    return ds.rio.write_crs(crs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Path to the model root folder, e.g. .../1_TI")
    parser.add_argument("--model-name", required=True, help="Short model label, e.g. TI, EB, iSnobal. Must match the '<name>_Task<N>' subfolder prefix.")
    parser.add_argument("--tasks", default="1,2", help="Comma-separated task numbers to process (default: 1,2)")
    parser.add_argument("--depth-factor", type=float, default=1.0, help="Multiply raw depth pixel values by this to get meters (default: 1.0, no conversion)")
    parser.add_argument("--swe-factor", type=float, default=1.0, help="Multiply raw SWE pixel values by this to get mm (default: 1.0, no conversion)")
    parser.add_argument("--nodata", type=float, default=-9999, help="Nodata sentinel written into output netCDFs (default: -9999)")
    parser.add_argument("--date-regex", default=DEFAULT_DATE_REGEX, help="Regex with 3 capture groups (year, month, day) applied to each filename (default: %(default)s)")
    parser.add_argument("--water-year-start-month", type=int, default=10, help="Month (1-12) that starts a water year (default: 10, i.e. Oct 1)")
    parser.add_argument("--crs", default="EPSG:32611", help="CRS hard-coded onto the output netCDFs (default: EPSG:32611)")
    parser.add_argument("--output-dir", default=None, help="Where to write netCDFs. Default: inside each <model>_Task<N> folder, matching 4_SM's convention.")
    parser.add_argument("--pad-mismatched-dates", action="store_true", help="If a date has SWE but no SD (or vice versa), keep it and fill the missing variable with nodata, instead of dropping it (default: drop, keep only dates present in both).")
    parser.add_argument("--dry-run", action="store_true", help="Report file counts/date ranges per water year without writing any netCDF.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tasks = [int(t) for t in args.tasks.split(",")]

    for task in tasks:
        task_dir = find_task_dir(model_dir, args.model_name, task)
        if task_dir is None:
            print(f"WARNING: no '{args.model_name}_Task{task}' folder found under {model_dir}, skipping.")
            continue

        print(f"\n=== {args.model_name} Task{task} ({task_dir}) ===")
        buckets = collect_files(task_dir, args.date_regex, args.water_year_start_month)
        if not buckets:
            print("  No matching files found.")
            continue

        out_dir = Path(args.output_dir) if args.output_dir else task_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        wrote_any = False
        for wy in sorted(buckets):
            depth_paths = buckets[wy]["SD"]
            swe_paths = buckets[wy]["SWE"]
            sd_range = (min(depth_paths).date(), max(depth_paths).date()) if depth_paths else None
            swe_range = (min(swe_paths).date(), max(swe_paths).date()) if swe_paths else None
            print(f"  WY{wy % 100:02d}: SD={len(depth_paths)} file(s) {sd_range}, SWE={len(swe_paths)} file(s) {swe_range}")

            if args.dry_run:
                continue

            ds = build_dataset(depth_paths, swe_paths, args.nodata, args.depth_factor, args.swe_factor, args.crs, args.pad_mismatched_dates)
            out_path = out_dir / f"{args.model_name}_WY{wy % 100:02d}_Task{task}.nc"
            # Set encoding on each DataArray directly rather than passing an
            # encoding= dict to to_netcdf(): with explicit per-variable dicts
            # for *every* data variable, to_netcdf() silently breaks the
            # grid_mapping link needed for ds.rio.crs to resolve on reopen
            # (verified: a single-variable encoding dict is fine, but two+
            # is not). Setting .encoding directly avoids that path entirely.
            for var in ("SD", "SWE"):
                ds[var].encoding.update({"zlib": True, "complevel": 4, "_FillValue": args.nodata})
            ds.to_netcdf(out_path)
            print(f"    -> wrote {out_path}")
            wrote_any = True

        # Drop a copy of this script and the exact command that produced these
        # netCDFs into the output folder, so anyone looking at the files later
        # can see how they were built (mirrors 4_SM/Task*/split_sd_swe.py).
        if wrote_any:
            this_script = Path(__file__).resolve()
            shutil.copy2(this_script, out_dir / this_script.name)
            command_log = out_dir / "build_model_netcdf_command.txt"
            command_log.write_text(
                f"Generated {out_dir.name}'s netCDF(s) with:\n\n"
                f"python {this_script.name} {' '.join(sys.argv[1:])}\n"
            )
            print(f"    -> copied {this_script.name} and build_model_netcdf_command.txt to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Search and download all Earthdata granules over an AOI shapefile via NASA Harmony.

Credentials (per https://harmony.earthdata.nasa.gov/docs#passing-credentials-in-code):
  - Default: ~/.netrc with
        machine urs.earthdata.nasa.gov login <user> password <pass>
    (chmod 0600 ~/.netrc)
  - Or set EDL_USERNAME / EDL_PASSWORD, or EDL_TOKEN (bearer token) env vars.

Example:
  python harmony_aoi_download.py aoi.shp C1234208438-POCLOUD \
      --start 2024-01-01 --end 2024-03-31 --out ./data
"""
import argparse
import os
import tempfile
from datetime import datetime

import geopandas as gpd
from harmony import BBox, CapabilitiesRequest, Client, Collection, Request


def make_client():
    """Client using token, username/password, or .netrc (in that order)."""
    if os.getenv("EDL_TOKEN"):
        return Client(token=os.environ["EDL_TOKEN"])
    if os.getenv("EDL_USERNAME") and os.getenv("EDL_PASSWORD"):
        return Client(auth=(os.environ["EDL_USERNAME"], os.environ["EDL_PASSWORD"]))
    return Client()  # falls back to ~/.netrc


def aoi_to_geojson(shp_path):
    """Reproject AOI to WGS84, dissolve to one geometry, write a temp GeoJSON."""
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    aoi = gdf.dissolve()[["geometry"]]
    out = os.path.join(tempfile.mkdtemp(), "aoi.geojson")
    aoi.to_file(out, driver="GeoJSON")
    return out, aoi.total_bounds  # (minx, miny, maxx, maxy)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("shapefile", help="AOI shapefile (.shp)")
    p.add_argument("collection", help="CMR collection concept ID or short name")
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument("--out", default="./harmony_downloads", help="Output directory")
    p.add_argument("--max-results", type=int, help="Limit number of granules (for testing)")
    args = p.parse_args()

    client = make_client()
    geojson, (w, s, e, n) = aoi_to_geojson(args.shapefile)

    # Check what this collection supports so we don't ask for an unsupported subset.
    caps = client.submit(CapabilitiesRequest(collection_id=args.collection))
    summary = caps.get("summary", {}).get("subsetting", {})
    shape_ok = summary.get("shape", caps.get("shapeSubset", False))
    bbox_ok = summary.get("bbox", caps.get("bboxSubset", False))

    req_kwargs = dict(
        collection=Collection(id=args.collection),
        skip_preview=True,          # don't auto-pause large (>300 granule) jobs
        ignore_errors=True,
    )
    if shape_ok:
        req_kwargs["shape"] = geojson
        print("Using shapefile (polygon) subsetting.")
    elif bbox_ok:
        req_kwargs["spatial"] = BBox(w, s, e, n)
        print("Shape subsetting not supported; using AOI bounding box.")
    else:
        print("Collection has no spatial subsetting in Harmony; "
              "request will return whole granules (no AOI filter applied by a service).")

    if args.start or args.end:
        req_kwargs["temporal"] = {
            k: datetime.fromisoformat(v)
            for k, v in (("start", args.start), ("stop", args.end)) if v
        }
    if args.max_results:
        req_kwargs["max_results"] = args.max_results

    request = Request(**req_kwargs)
    if not request.is_valid():
        raise SystemExit(f"Invalid request: {request.error_messages()}")

    job_id = client.submit(request)
    print(f"Submitted job {job_id}\n  Monitor: https://harmony.earthdata.nasa.gov/workflow-ui/{job_id}")
    client.wait_for_processing(job_id, show_progress=True)

    os.makedirs(args.out, exist_ok=True)
    futures = client.download_all(job_id, directory=args.out, overwrite=False)
    files = [f.result() for f in futures]
    print(f"Downloaded {len(files)} files to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
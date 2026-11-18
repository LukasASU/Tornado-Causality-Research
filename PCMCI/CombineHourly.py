#!/usr/bin/env python3
"""
STEP 1 (optional, only for the hourly ERA5 data): combine the per-year NetCDF files
into two CSV files. No derived variables, no thresholds, no monthly averaging.

Input  (Data/era5_round2/):
    ERA5_hourly_500hPa_winds_<year>.nc       u, v at 500 hPa      (pressure-level data)
    ERA5_hourly_cape_10mwinds_<year>.nc      cape, u10, v10       (surface data)

Output (Data/):
    era5_pressure_winds.csv   time, u500, v500
    era5_surface.csv          time, cape, u10, v10
    (each with a .meta.json recording the box that was used)

The only thing done to the data is a spatial average over the box (area-weighted by
cos(latitude)) at each time step. The raw grid has hundreds of millions of rows for 43
years, which is too large for a CSV; everything else (shear, thresholds, monthly values)
happens in preprocess_data.py.

Usage (use the SAME --box as the rest of the project):
    python combine_hourly.py --box 43 26 -106 -94
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_BOX = (43.0, 26.0, -106.0, -94.0)   # N, S, W, E

# output file -> (file pattern, variables to read, column names in the CSV)
GROUPS = {
    "era5_pressure_winds.csv": ("*500hPa_winds*.nc", ("u", "v"), ("u500", "v500")),
    "era5_surface.csv": ("*cape_10mwinds*.nc", ("cape", "u10", "v10"),
                         ("cape", "u10", "v10")),
}


def area_weighted_mean(values, lat):
    """values: (time, lat, lon) array -> (time,) cos(lat)-weighted mean, NaNs ignored."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[None, :, None]
    w = np.broadcast_to(w, values.shape) * np.isfinite(values)
    norm = w.sum(axis=(1, 2))
    norm = np.where(norm == 0, np.nan, norm)
    return (np.nan_to_num(values) * w).sum(axis=(1, 2)) / norm


def read_file(path, variables, columns, box):
    """One NetCDF file -> DataFrame of regional means (index = time)."""
    import xarray as xr
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    north, south, west, east = box
    lat_desc = bool(ds["latitude"][0] > ds["latitude"][-1])
    lat_sl = slice(north, south) if lat_desc else slice(south, north)
    ds = ds.sel({"latitude": lat_sl, "longitude": slice(west, east)})
    out = {}
    for var, col in zip(variables, columns):
        if var not in ds.data_vars:
            sys.exit(f"{path}: variable '{var}' not found; file has {list(ds.data_vars)}")
        da = ds[var]
        if "pressure_level" in da.dims:
            da = da.isel(pressure_level=0)       # only 500 hPa was downloaded
        da = da.transpose(tdim, "latitude", "longitude")
        if da.size == 0:
            sys.exit(f"{path}: no grid cells inside the box {box}")
        out[col] = area_weighted_mean(da.values.astype(float), da["latitude"].values)
    return pd.DataFrame(out, index=pd.DatetimeIndex(ds[tdim].values))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=os.path.join("Data", "era5_round2"))
    ap.add_argument("--out-dir", default="Data")
    ap.add_argument("--box", nargs=4, type=float, metavar=("N", "S", "W", "E"),
                    default=list(DEFAULT_BOX))
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    for out_name, (pattern, variables, columns) in GROUPS.items():
        files = sorted(glob.glob(os.path.join(a.in_dir, pattern)))
        if not files:
            print(f"No files matching {pattern} in {a.in_dir}; skipping {out_name}")
            continue
        print(f"\n{out_name}: combining {len(files)} files")
        frames = []
        for i, f in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}")
            frames.append(read_file(f, variables, columns, tuple(a.box)))
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        df.index.name = "time"
        path = os.path.join(a.out_dir, out_name)
        df.to_csv(path)
        with open(os.path.splitext(path)[0] + ".meta.json", "w") as fh:
            json.dump({"box_NSWE": a.box}, fh)
        print(f"  wrote {path}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print("\nNext: python preprocess_data.py --hourly ...")


if __name__ == "__main__":
    main()
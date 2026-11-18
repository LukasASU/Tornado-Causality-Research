#!/usr/bin/env python3
"""
STEP 2: preprocessing. Raw data files -> one clean monthly table per outcome.

This script does everything that happens BEFORE the causal algorithm:
    load each source -> put on a common monthly index -> merge -> remove the seasonal
    cycle and linear trend (standardized anomalies) -> save.
It does not import tigramite. The next script (pcmci.py) reads what this one writes.

Expected files in ./Data :
    1950-2025_SPC.csv                    SPC "actual tornadoes" CSV
    ERA5_core_monthly.nc                 ERA5 monthly means: cape, d2m, t2m
    ERA5_Volumetric_soil_layer_1.nc      (with --extra soil)
    ERA5_total_precipitation.nc          (with --extra precip)
    ERDDAP.csv                           ERSST v5 'ssta' for the Gulf box
    NOAA_SST.txt                         monthly Nino indices
    direct_loss_aggregated_output_*.csv  SHELDUS county x year x month losses
    era5_pressure_winds.csv, era5_surface.csv   from combine_hourly.py (with --hourly)

Run (one line in PowerShell):
    python preprocess_data.py --outcome both --era5-file ERA5_core_monthly.nc --extra soil precip --hourly --box 43 26 -106 -94

Writes to ./output:
    merged_monthly_<outcome>.csv     raw monthly values (before anomalies)
    preprocessed_<outcome>.csv       standardized anomalies: this is what PCMCI reads
    preprocessed_<outcome>.meta.json states / box / dates, used for graph captions
    qa_series_<outcome>.png          quick look at every series
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
STATES = ["TX", "OK", "KS", "NE"]
# ERA5 averaging box (matches the box used for the ERA5 request)
BOX = dict(north=43.0, south=26.0, west=-106.0, east=-94.0)

STATE_NAMES = {
    "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS",
    "CA": "CALIFORNIA", "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE",
    "FL": "FLORIDA", "GA": "GEORGIA", "HI": "HAWAII", "ID": "IDAHO",
    "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA", "KS": "KANSAS",
    "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND",
    "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI",
    "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA",
    "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK",
    "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA",
    "OR": "OREGON", "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA", "TN": "TENNESSEE", "TX": "TEXAS", "UT": "UTAH",
    "VT": "VERMONT", "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST VIRGINIA",
    "WI": "WISCONSIN", "WY": "WYOMING",
}


# --------------------------------------------------------------------------
# Loaders: each returns a monthly pandas Series / DataFrame indexed by month start
# --------------------------------------------------------------------------
def month_index(start, end):
    return pd.date_range(start, end, freq="MS")


def load_spc(path, states, start, end, min_mag=1):
    """log(1 + number of days per month with an EF>=min_mag tornado in `states`)."""
    df = pd.read_csv(path, usecols=["yr", "mo", "dy", "st", "mag"])
    df = df[df["st"].isin(states) & (df["mag"] >= min_mag)].copy()
    df["date"] = pd.to_datetime(pd.DataFrame({"year": df["yr"], "month": df["mo"],
                                              "day": df["dy"]}))
    days = df.drop_duplicates("date")
    s = days.groupby(days["date"].dt.to_period("M")).size()
    s.index = pd.PeriodIndex(s.index).to_timestamp()
    s = s.reindex(month_index(start, end), fill_value=0).astype(float)
    return s.apply(np.log1p).rename("tornado_days")


def load_enso(path, start, end):
    """Niño 3.4 anomaly = last column of the NOAA monthly Nino-indices table."""
    df = pd.read_csv(path, sep=r"\s+")
    date = pd.to_datetime(pd.DataFrame({"year": df["YR"], "month": df["MON"], "day": 1}))
    s = pd.Series(df.iloc[:, -1].astype(float).to_numpy(), index=date)
    s = s.mask(s <= -99)
    return s.reindex(month_index(start, end)).rename("enso")


def load_gulf_sst(path, start, end):
    """Monthly mean of ERSST anomaly over the Gulf box (land cells are NaN and ignored)."""
    df = pd.read_csv(path, skiprows=[1])  # row 1 is the units row
    df["ssta"] = pd.to_numeric(df["ssta"], errors="coerce")
    t = pd.to_datetime(df["time"])
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    df["month"] = t.dt.to_period("M").dt.to_timestamp()
    s = df.groupby("month")["ssta"].mean()
    return s.reindex(month_index(start, end)).rename("gulf_sst")


def load_era5(path, start, end, box=None):
    """Area-weighted regional monthly means of CAPE, 2 m dewpoint, 2 m temperature.
    Subsets the box BEFORE loading, so the large file is never fully read."""
    import xarray as xr
    box = box or BOX
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_desc = bool(ds["latitude"][0] > ds["latitude"][-1])
    lat_sl = (slice(box["north"], box["south"]) if lat_desc
              else slice(box["south"], box["north"]))
    variables = ["cape", "d2m", "t2m"]
    sub = ds[variables].sel({tdim: slice(start, end), "latitude": lat_sl,
                             "longitude": slice(box["west"], box["east"])})
    weights = sub["latitude"].copy(data=np.cos(np.deg2rad(sub["latitude"].values)))
    mean = sub.weighted(weights).mean(dim=["latitude", "longitude"])
    df = mean.to_dataframe()[variables]
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    df = df.rename(columns={"cape": "cape", "d2m": "dewpoint", "t2m": "t2m"})
    return df.reindex(month_index(start, end))


# optional extra ERA5 monthly variables: key -> (file, candidate variable names, column name)
EXTRA_ERA5 = {
    "soil": ("ERA5_Volumetric_soil_layer_1.nc", ("swvl1",), "soil_moisture"),
    "precip": ("ERA5_total_precipitation.nc", ("tp",), "precip"),
}


def load_era5_monthly(path, start, end, box, candidates, name):
    """Area-weighted regional monthly mean of ONE variable from an ERA5 monthly file."""
    import xarray as xr
    ds = xr.open_dataset(path)
    var = next((v for v in candidates if v in ds.data_vars), None)
    if var is None:
        if len(ds.data_vars) == 1:
            var = list(ds.data_vars)[0]
        else:
            sys.exit(f"{path}: expected one of {candidates}, found {list(ds.data_vars)}")
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_desc = bool(ds["latitude"][0] > ds["latitude"][-1])
    lat_sl = (slice(box["north"], box["south"]) if lat_desc
              else slice(box["south"], box["north"]))
    sub = ds[var].sel({tdim: slice(start, end), "latitude": lat_sl,
                       "longitude": slice(box["west"], box["east"])})
    if sub.size == 0:
        sys.exit(f"{path}: no data inside the box {box}. Check the area used when "
                 f"downloading (North, West, South, East = 43, -107, 25, -84).")
    weights = sub["latitude"].copy(data=np.cos(np.deg2rad(sub["latitude"].values)))
    s = sub.weighted(weights).mean(dim=["latitude", "longitude"]).to_series()
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    return s.reindex(month_index(start, end)).rename(name)


def load_sheldus(path, states, start, end):
    """log(1 + inflation-adjusted property+crop damage) from tornadoes, per month."""
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    if "year" not in low:
        sys.exit(
            "SHELDUS.csv has no 'Year' column, so it cannot be turned into a monthly\n"
            "time series (it looks aggregated by calendar month only).\n"
            "Re-download with aggregation by county (or state) + YEAR AND MONTH.")
    names = [STATE_NAMES[s] for s in states]
    df = df[df[low["statename"]].str.upper().isin(names)]
    if "hazard" in low:
        df = df[df[low["hazard"]].astype(str).str.contains("tornado", case=False)]
    else:
        print("Note: SHELDUS.csv has no Hazard column; assuming the query was filtered "
              "to Tornado only.")
    loss_cols = [c for c in df.columns
                 if c.startswith("PropertyDmg(ADJ") or c.startswith("CropDmg(ADJ")]
    if not loss_cols:
        print("WARNING: no inflation-adjusted columns found; using raw damage columns.")
        loss_cols = [c for c in ("PropertyDmg", "CropDmg") if c in df.columns]
    df = df.assign(loss=df[loss_cols].sum(axis=1))
    g = df.groupby([low["year"], low["month"]])["loss"].sum().reset_index()
    date = pd.to_datetime(pd.DataFrame({"year": g[low["year"]], "month": g[low["month"]],
                                        "day": 1}))
    s = pd.Series(g["loss"].to_numpy(), index=date)
    s = s.reindex(month_index(start, end), fill_value=0.0)  # absent month = no losses
    return s.apply(np.log1p).rename("sheldus_loss")


def find_sheldus(data_dir, explicit=None):
    """Use --sheldus-file if given, else SHELDUS.csv, else the SHELDUS download's own name
    (direct_loss_aggregated_output_*.csv). The insured_crop_* file is not used."""
    import glob
    if explicit:
        return explicit
    plain = os.path.join(data_dir, "SHELDUS.csv")
    if os.path.exists(plain):
        return plain
    hits = sorted(glob.glob(os.path.join(data_dir, "direct_loss_aggregated_output_*.csv")))
    if not hits:
        sys.exit(f"No SHELDUS direct-loss file found in {data_dir}/")
    return hits[0]



# --------------------------------------------------------------------------
# Hourly ERA5 (the two CSVs from combine_hourly.py) -> monthly columns
# --------------------------------------------------------------------------
HOURLY_VARS = ["shear", "cape_hourly", "cape_x_shear", "favorable_days"]


def check_box(path, box):
    """Warn if a combine_hourly.py CSV was made for a different box than this run."""
    meta = os.path.splitext(path)[0] + ".meta.json"
    if not os.path.exists(meta):
        return
    saved = json.load(open(meta)).get("box_NSWE")
    mine = [box["north"], box["south"], box["west"], box["east"]]
    if saved and any(abs(x - y) > 1e-6 for x, y in zip(saved, mine)):
        print(f"WARNING: {os.path.basename(path)} was built for box (N,S,W,E) {saved} "
              f"but this run uses {mine}. Re-run combine_hourly.py with the same --box.")


def hourly_monthly(wind_csv, surface_csv, cols, start, end, box, cape_thr, shear_thr):
    """Derive shear etc. from the raw regional-mean CSVs, then take monthly values.

    shear           = |wind(500 hPa) - wind(10 m)| in m/s (proxy for deep-layer shear)
    cape_hourly     = CAPE at the downloaded hours
    cape_x_shear    = CAPE * shear
    favorable_days  = days on which regional-mean CAPE >= cape_thr AND shear >= shear_thr
                      (log1p applied, like the other counts)
    """
    for p in (wind_csv, surface_csv):
        if not os.path.exists(p):
            sys.exit(f"{p} not found. Run combine_hourly.py first.")
        check_box(p, box)
    w = pd.read_csv(wind_csv, parse_dates=["time"], index_col="time")
    s = pd.read_csv(surface_csv, parse_dates=["time"], index_col="time")
    h = w.join(s, how="inner")
    h["shear"] = np.hypot(h["u500"] - h["u10"], h["v500"] - h["v10"])
    h["cape_hourly"] = h["cape"]
    h["cape_x_shear"] = h["cape"] * h["shear"]
    fav = (h["cape"] >= cape_thr) & (h["shear"] >= shear_thr)
    times = pd.DatetimeIndex(h.index)
    daily_fav = fav.groupby(times.normalize()).any()
    month = times.to_period("M")
    monthly = h[["shear", "cape_hourly", "cape_x_shear"]].groupby(month).mean()
    monthly["favorable_days"] = (daily_fav.groupby(pd.DatetimeIndex(daily_fav.index).to_period("M"))
                                 .sum().astype(float).apply(np.log1p))
    monthly.index = pd.PeriodIndex(monthly.index).to_timestamp()
    return monthly[cols].reindex(month_index(start, end))


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def to_anomalies(df):
    """Per-calendar-month standardized anomalies, then remove a linear trend."""
    out = df.copy().astype(float)
    month = out.index.month
    for c in out.columns:
        g = out[c].groupby(month)
        sd = g.transform("std").replace(0, np.nan)
        out[c] = (out[c] - g.transform("mean")) / sd
    t = np.arange(len(out), dtype=float)
    for c in out.columns:
        y = out[c].to_numpy()
        ok = np.isfinite(y)
        coef = np.polyfit(t[ok], y[ok], 1)
        out[c] = y - np.polyval(coef, t)
    return out


def stationarity_report(df):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("(statsmodels not installed; skipping ADF stationarity test)")
        return
    print("\nADF stationarity test (small p-value = stationary):")
    for c in df.columns:
        print(f"  {c:14s} p = {adfuller(df[c].dropna())[1]:.4f}")


def save_qa_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 1.8 * len(df.columns)),
                             sharex=True)
    for ax, c in zip(np.atleast_1d(axes), df.columns):
        ax.plot(df.index, df[c], lw=0.8)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../Data")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--sheldus-file", default=None,
                    help="path to the SHELDUS direct-loss CSV (auto-detected by default)")
    ap.add_argument("--outcome", choices=["spc", "sheldus", "both"], default="spc")
    ap.add_argument("--states", nargs="+", default=STATES)
    ap.add_argument("--box", nargs=4, type=float, metavar=("N", "S", "W", "E"), default=None,
                    help="ERA5 averaging box: north south west east (west negative). "
                         "Should cover the same area as --states.")
    ap.add_argument("--start", default="1982-01")   # the ENSO file starts in 1982
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--era5-file", default="ERA5_core_monthly.nc",
                    help="monthly ERA5 file with cape/d2m/t2m (inside --data-dir)")
    ap.add_argument("--extra", nargs="*", choices=sorted(EXTRA_ERA5), default=[],
                    help="extra monthly ERA5 variables: soil precip")
    ap.add_argument("--hourly", action="store_true",
                    help="add columns derived from the hourly CSVs (combine_hourly.py)")
    ap.add_argument("--hourly-vars", nargs="*", choices=HOURLY_VARS,
                    default=["shear", "favorable_days"])
    ap.add_argument("--cape-thr", type=float, default=500.0, help="J/kg, for favorable_days")
    ap.add_argument("--shear-thr", type=float, default=15.0, help="m/s, for favorable_days")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    d = lambda f: os.path.join(a.data_dir, f)
    box = dict(zip(("north", "south", "west", "east"), a.box)) if a.box else BOX
    print(f"States: {', '.join(a.states)} | ERA5 box (N,S,W,E): "
          f"{box['north']}, {box['south']}, {box['west']}, {box['east']}")

    print("Loading predictors...")
    pieces = [
        load_enso(d("NOAA_SST.txt"), a.start, a.end),
        load_gulf_sst(d("ERDDAP.csv"), a.start, a.end),
        load_era5(d(a.era5_file), a.start, a.end, box),
    ]
    for key in a.extra:
        fname, cands, col = EXTRA_ERA5[key]
        pieces.append(load_era5_monthly(d(fname), a.start, a.end, box, cands, col))
    if a.hourly:
        pieces.append(hourly_monthly(d("era5_pressure_winds.csv"), d("era5_surface.csv"),
                                     a.hourly_vars, a.start, a.end, box,
                                     a.cape_thr, a.shear_thr))
    predictors = pd.concat(pieces, axis=1)

    outcomes = {}
    if a.outcome in ("spc", "both"):
        outcomes["tornado_days"] = load_spc(d("1950-2025_SPC.csv"), a.states, a.start, a.end)
    if a.outcome in ("sheldus", "both"):
        outcomes["sheldus_loss"] = load_sheldus(find_sheldus(a.data_dir, a.sheldus_file),
                                                a.states, a.start, a.end)

    for name, series in outcomes.items():
        df = pd.concat([predictors, series], axis=1)
        print(f"\n=== Outcome: {name} ===")
        print("Non-missing months per variable:")
        print(df.notna().sum().to_string())
        n0 = len(df)
        df = df.dropna()
        if len(df) < n0:
            print(f"Dropped {n0 - len(df)} months with missing values; {len(df)} remain.")
        df.to_csv(os.path.join(a.out_dir, f"merged_monthly_{name}.csv"))
        print("\nCorrelation matrix (raw monthly values):")
        print(df.corr().round(2).to_string())
        save_qa_plot(df, os.path.join(a.out_dir, f"qa_series_{name}.png"))

        an = to_anomalies(df).dropna()
        an.index.name = "date"
        an.to_csv(os.path.join(a.out_dir, f"preprocessed_{name}.csv"))
        with open(os.path.join(a.out_dir, f"preprocessed_{name}.meta.json"), "w") as f:
            json.dump({"states": a.states,
                       "box_NSWE": [box["north"], box["south"], box["west"], box["east"]],
                       "start": a.start, "end": a.end}, f)
        stationarity_report(an)
        print(f"Wrote preprocessed_{name}.csv ({len(an)} months x {an.shape[1]} variables)")

    print(f"\nDone. Next: python pcmci.py --outcome {a.outcome}")


if __name__ == "__main__":
    main()
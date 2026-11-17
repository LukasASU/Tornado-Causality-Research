#!/usr/bin/env python3
"""
Tornado causality pipeline: load -> merge -> preprocess -> PCMCI+.

Expected files in ./Data :
    1950-2025_SPC.csv   SPC "actual tornadoes" CSV
    ERA5.nc             ERA5 monthly means: cape, d2m, t2m
    ERDDAP.csv          ERSST v5 'ssta' for the Gulf box (ERDDAP CSV, with units row)
    NOAA_SST.txt        monthly Nino indices (YR MON NINO1+2 ANOM ... NINO3.4 ANOM)
    direct_loss_aggregated_output_*.csv  SHELDUS county x year x month losses (optional;
                        a file named SHELDUS.csv also works)

Run (PowerShell, one line):
    python pcmci_pipeline.py --outcome spc
    python pcmci_pipeline.py --outcome sheldus
    python pcmci_pipeline.py --outcome both --tau-max 6

Outputs go to ./output
"""
import argparse
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
    weights = np.cos(np.deg2rad(sub["latitude"]))
    mean = sub.weighted(weights).mean(dim=["latitude", "longitude"])
    df = mean.to_dataframe()[variables]
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
    df = df.rename(columns={"cape": "cape", "d2m": "dewpoint", "t2m": "t2m"})
    return df.reindex(month_index(start, end))


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


# --------------------------------------------------------------------------
# PCMCI+
# --------------------------------------------------------------------------
def run_pcmciplus(df, tau_max, alpha):
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        sys.exit("tigramite is not installed. Run: python -m pip install tigramite joblib")
    dataframe = pp.DataFrame(df.to_numpy(dtype=float), var_names=list(df.columns))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    # PCMCI+ uses a single significance level (pc_alpha) for all of its tests.
    return pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)


def extract_links(res, names, tau_max):
    g, v, p = res["graph"], res["val_matrix"], res["p_matrix"]
    n = len(names)
    rows = []
    for j in range(n):
        for i in range(n):
            for tau in range(tau_max + 1):
                s = g[i, j, tau]
                if s == "":
                    continue
                if tau == 0:
                    if i == j or s == "<--":  # '<--' is reported from the other side
                        continue
                    if s in ("o-o", "x-x", "x-o", "o-x") and i > j:
                        continue
                rows.append(dict(source=names[i], target=names[j], lag=tau, mark=s,
                                 strength=float(v[i, j, tau]), p_value=float(p[i, j, tau])))
    return pd.DataFrame(rows)


# Node labels shown in the graph (abbreviations spelled out; "\n" = line break)
DISPLAY_NAMES = {
    "enso": "El Niño–Southern\nOscillation (ENSO)",
    "gulf_sst": "Gulf of Mexico\nSea Surface Temp.",
    "cape": "Convective Available\nPotential Energy (CAPE)",
    "dewpoint": "2 m Dewpoint\nTemperature",
    "t2m": "2 m Air\nTemperature",
    "tornado_days": "Tornado Days\n(EF1+ per month)",
    "sheldus_loss": "Tornado Losses\n(SHELDUS)",
}
# Detail text for the legend under the graph
VAR_DETAILS = {
    "enso": "Niño 3.4 sea-surface-temperature anomaly, tropical Pacific",
    "gulf_sst": "ERSST v5 anomaly, averaged over the Gulf of Mexico box",
    "cape": "ERA5 regional monthly mean; storm instability",
    "dewpoint": "ERA5 regional monthly mean; near-surface moisture",
    "t2m": "ERA5 regional monthly mean",
    "tornado_days": "log(1 + days per month with an EF1+ tornado), {states}, SPC",
    "sheldus_loss": "log(1 + inflation-adjusted property+crop losses), {states}, SHELDUS",
}
HOW_TO_READ = [
    "Arrowhead  =  directed link (cause -> effect)",
    "Circle on an end  =  direction not determined",
    "Number on an arrow  =  lag in months",
    "No number  =  same-month link",
    "Red = positive, blue = negative relationship;",
    "   paler / thinner = weaker",
    "Node color (left bar) = how strongly the",
    "   variable persists from month to month",
    "Link color (right bar) = link strength",
]


def draw_graph_png(res, labels, raw_path):
    """Render tigramite's graph to a PNG without clipping the colorbar labels."""
    import matplotlib.pyplot as plt
    from tigramite import plotting as tp
    kw = dict(graph=res["graph"], val_matrix=res["val_matrix"], var_names=labels)
    try:
        # draw on our own figure so we can save with bbox_inches="tight"
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, frame_on=False)
        tp.plot_graph(fig_ax=(fig, ax), link_colorbar_label="Link strength (MCI)",
                      node_colorbar_label="Persistence (auto-MCI)", **kw)
        fig.savefig(raw_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    except TypeError:  # older tigramite without these options
        plt.close("all")
        tp.plot_graph(save_name=raw_path, **kw)


def save_graph(res, names, path, states):
    """Save the graph with spelled-out node names plus a legend panel underneath."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        raw = path.replace(".png", "_raw.png")
        labels = [DISPLAY_NAMES.get(n, n) for n in names]
        draw_graph_png(res, labels, raw)
        plt.close("all")
        img = plt.imread(raw)
        fig = plt.figure(figsize=(11, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25])
        ax = fig.add_subplot(gs[0])
        ax.imshow(img)
        ax.axis("off")
        lax = fig.add_subplot(gs[1])
        lax.axis("off")
        lax.text(0.0, 1.0, "Variables", fontsize=11, fontweight="bold", va="top")
        st = "/".join(states)
        lines = [DISPLAY_NAMES.get(n, n).replace("\n", " ") + ": "
                 + VAR_DETAILS.get(n, "").format(states=st) for n in names]
        lax.text(0.0, 0.88, "\n\n".join(lines), fontsize=7.5, va="top", linespacing=1.1)
        lax.text(0.66, 1.0, "How to read the graph", fontsize=11, fontweight="bold",
                 va="top")
        lax.text(0.66, 0.88, "\n".join(HOW_TO_READ), fontsize=8.5, va="top",
                 linespacing=1.5)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        os.remove(raw)
    except Exception as e:  # plotting should never kill a finished run
        print(f"(could not save graph image: {e})")


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
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--sheldus-file", default=None,
                    help="path to the SHELDUS direct-loss CSV (auto-detected by default)")
    ap.add_argument("--outcome", choices=["spc", "sheldus", "both"], default="spc")
    ap.add_argument("--states", nargs="+", default=STATES)
    # ENSO file starts in 1982, so that is the first common month
    ap.add_argument("--box", nargs=4, type=float, metavar=("N", "S", "W", "E"),
                    default=None,
                    help="ERA5 averaging box: north south west east (degrees, west negative). "
                         "Should cover the same area as --states.")
    ap.add_argument("--start", default="1982-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--tau-max", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--drop", nargs="*", default=[],
                    help="predictor columns to leave out, e.g. --drop t2m")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    d = lambda f: os.path.join(a.data_dir, f)

    box = dict(zip(("north", "south", "west", "east"), a.box)) if a.box else BOX
    print(f"States: {', '.join(a.states)} | ERA5 box (N,S,W,E): "
          f"{box['north']}, {box['south']}, {box['west']}, {box['east']}")
    print("Loading predictors...")
    predictors = pd.concat([
        load_enso(d("NOAA_SST.txt"), a.start, a.end),
        load_gulf_sst(d("ERDDAP.csv"), a.start, a.end),
        load_era5(d("ERA5.nc"), a.start, a.end, box),
    ], axis=1).drop(columns=a.drop, errors="ignore")

    outcomes = {}
    if a.outcome in ("spc", "both"):
        outcomes["tornado_days"] = load_spc(d("1950-2025_SPC.csv"), a.states,
                                            a.start, a.end)
    if a.outcome in ("sheldus", "both"):
        outcomes["sheldus_loss"] = load_sheldus(find_sheldus(a.data_dir, a.sheldus_file), a.states,
                                                a.start, a.end)

    for name, series in outcomes.items():
        df = pd.concat([predictors, series], axis=1)
        print(f"\n=== Run: outcome = {name} ===")
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
        an.to_csv(os.path.join(a.out_dir, f"preprocessed_{name}.csv"))
        stationarity_report(an)

        print(f"\nRunning PCMCI+ on {len(an)} months x {an.shape[1]} variables "
              f"(tau_max={a.tau_max}, alpha={a.alpha})")
        res = run_pcmciplus(an, a.tau_max, a.alpha)
        links = extract_links(res, list(an.columns), a.tau_max)
        if links.empty:
            print("No significant links.")
        else:
            links = links.sort_values(["target", "lag", "source"]).reset_index(drop=True)
            print(links.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            links.to_csv(os.path.join(a.out_dir, f"links_{name}.csv"), index=False)
        save_graph(res, list(an.columns), os.path.join(a.out_dir, f"graph_{name}.png"),
                   a.states)
        print("(mark '-->' = directed; 'o-o' = same-month link whose direction is undetermined)")

    print(f"\nDone. Results in ./{a.out_dir}")


if __name__ == "__main__":
    main()
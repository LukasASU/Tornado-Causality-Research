#!/usr/bin/env python3
"""
PCMCI+ proof of concept for tornado causality.

Install once:   pip install tigramite pandas numpy scipy

Three modes
-----------
1) Synthetic validation (known causal graph; run this FIRST):
       python pcmci_poc.py synthetic
       python pcmci_poc.py synthetic --seasonal      # also tests the preprocessing
       python pcmci_poc.py synthetic --shuffle       # negative control: expect ~0 links

2) Real data:
       python pcmci_poc.py real \
           --spc 1950-2024_actual_tornadoes.csv \
           --states TX OK KS NE \
           --series enso=nino34.ascii.txt nao=nao_monthly.csv \
           --start 1979-01 --end 2024-12

   --series takes name=path pairs. Each file is either
     * a whitespace table with YR and MON columns (e.g. CPC detrend.nino34.ascii.txt;
       the LAST column is used as the value), or
     * a CSV with columns  date,value  (date parseable, one row per month).

   Add --shuffle for the negative control on real data.
"""
import argparse
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Synthetic data with a known causal graph
# --------------------------------------------------------------------------
SYN_NAMES = ["driver", "shear", "moisture", "tornado"]
# (source, target, lag) -- lag 0 means contemporaneous
SYN_TRUTH = [
    (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1),  # autocorrelation
    (0, 1, 1),   # driver(t-1) -> shear(t)
    (0, 2, 2),   # driver(t-2) -> moisture(t)
    (1, 3, 0),   # shear(t)    -> tornado(t)   (contemporaneous)
    (2, 3, 1),   # moisture(t-1) -> tornado(t)
]


def make_synthetic(n=800, seed=0, seasonal=False):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 4))
    e = rng.normal(size=(n, 4))
    for t in range(2, n):
        x[t, 0] = 0.6 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.6 * x[t - 1, 0] + e[t, 1]
        x[t, 2] = 0.5 * x[t - 1, 2] + 0.5 * x[t - 2, 0] + e[t, 2]
        x[t, 3] = 0.3 * x[t - 1, 3] + 0.5 * x[t, 1] + 0.4 * x[t - 1, 2] + e[t, 3]
    idx = pd.date_range("1950-01-01", periods=n, freq="MS")
    if seasonal:
        month = idx.month.values
        tt = np.arange(n)
        for i in range(4):
            x[:, i] += 3 * np.sin(2 * np.pi * month / 12 + i) + 0.004 * tt * (i + 1)
    return pd.DataFrame(x, index=idx, columns=SYN_NAMES)


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


# --------------------------------------------------------------------------
# Real-data loaders
# --------------------------------------------------------------------------
def load_spc_tornado_days(path, states, start, end, min_mag=1):
    """Monthly count of distinct calendar days with >= 1 tornado of EF >= min_mag
    in the given states, from the SPC 'actual_tornadoes' CSV."""
    df = pd.read_csv(path)
    need = {"yr", "mo", "dy", "st", "mag"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"SPC file is missing columns {sorted(missing)}; found {list(df.columns)}")
    df = df[df["st"].isin(states) & (df["mag"] >= min_mag)].copy()
    df["date"] = pd.to_datetime(pd.DataFrame({"year": df["yr"], "month": df["mo"], "day": df["dy"]}))
    days = df.drop_duplicates("date")
    s = days.groupby(days["date"].dt.to_period("M")).size()
    s.index = pd.PeriodIndex(s.index).to_timestamp()
    full = pd.date_range(start, end, freq="MS")
    return s.reindex(full, fill_value=0).astype(float)


def load_monthly_series(path, start, end):
    """Whitespace table with YR/MON columns (last column = value), or CSV date,value."""
    try:
        df = pd.read_csv(path, sep=r"\s+")
    except Exception:
        df = None
    if df is not None and {"YR", "MON"} <= set(df.columns):
        date = pd.to_datetime(pd.DataFrame({"year": df["YR"], "month": df["MON"], "day": 1}))
        s = pd.Series(df.iloc[:, -1].astype(float).to_numpy(), index=date)
    else:
        df = pd.read_csv(path)
        if not {"date", "value"} <= set(df.columns):
            sys.exit(f"{path}: need YR/MON columns or CSV columns 'date,value'")
        date = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        s = pd.Series(df["value"].astype(float).to_numpy(), index=date)
    s = s.mask(s <= -99)  # common missing-value sentinels (-99.9, -9999)
    full = pd.date_range(start, end, freq="MS")
    return s.reindex(full)


# --------------------------------------------------------------------------
# PCMCI+
# --------------------------------------------------------------------------
def run_pcmciplus(df, tau_max, alpha):
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except ImportError:
        sys.exit("tigramite is not installed. Run:  pip install tigramite")
    dataframe = pp.DataFrame(df.to_numpy(dtype=float), var_names=list(df.columns))
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"),
                  verbosity=0)
    # PCMCI+ uses a single significance level (pc_alpha) for all of its tests.
    return pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha)


def extract_links(res, names, tau_max):
    """Return list of dicts for every significant link in the PCMCI+ graph."""
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
                rows.append(dict(source=names[i], target=names[j], lag=tau,
                                 mark=s, strength=float(v[i, j, tau]),
                                 p_value=float(p[i, j, tau])))
    return pd.DataFrame(rows)


def score_against_truth(links, names):
    """Precision/recall vs SYN_TRUTH. Contemporaneous links compared ignoring direction."""
    def key(a, b, lag):
        return (frozenset((a, b)), 0) if lag == 0 else ((a, b), lag)

    truth = {key(names[a], names[b], lag) for a, b, lag in SYN_TRUTH}
    found = {key(r.source, r.target, r.lag) for r in links.itertuples()}
    tp, fp, fn = truth & found, found - truth, truth - found
    print(f"\nTrue links: {len(truth)} | found: {len(found)} | "
          f"TP: {len(tp)} FP: {len(fp)} FN: {len(fn)}")
    if fp:
        print("  false positives:", sorted(map(str, fp)))
    if fn:
        print("  missed:", sorted(map(str, fn)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synthetic", "real"])
    ap.add_argument("--tau-max", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="significance level for PCMCI+ (smaller = fewer, stronger links)")
    ap.add_argument("--shuffle", action="store_true",
                    help="negative control: independently permute every column")
    ap.add_argument("--seasonal", action="store_true", help="synthetic: add season+trend")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spc", help="SPC *_actual_tornadoes.csv")
    ap.add_argument("--states", nargs="+", default=["TX", "OK", "KS", "NE"])
    ap.add_argument("--min-mag", type=int, default=1, help="minimum EF rating (1 = EF1+)")
    ap.add_argument("--series", nargs="*", default=[], help="name=path monthly series")
    ap.add_argument("--start", default="1979-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--out", default="links.csv")
    a = ap.parse_args()

    if a.mode == "synthetic":
        df = make_synthetic(seed=a.seed, seasonal=a.seasonal)
    else:
        if not a.spc:
            sys.exit("--spc is required in real mode")
        tornado = load_spc_tornado_days(a.spc, a.states, a.start, a.end, a.min_mag)
        cols: dict[str, pd.Series] = {"tornado_days": tornado.apply(np.log1p)}
        for item in a.series:
            name, _, path = item.partition("=")
            cols[name] = load_monthly_series(path, a.start, a.end)
        df = pd.DataFrame(cols)
        n_before = len(df)
        df = df.dropna()
        if len(df) < n_before:
            print(f"Dropped {n_before - len(df)} months with missing values.")

    df = to_anomalies(df).dropna()
    if a.shuffle:
        rng = np.random.default_rng(a.seed)
        df = pd.DataFrame({c: rng.permutation(df[c].to_numpy()) for c in df.columns},
                          index=df.index)
        print("NEGATIVE CONTROL: columns independently shuffled; expect ~no links.")

    print(f"Running PCMCI+ on {df.shape[0]} months x {df.shape[1]} variables "
          f"({', '.join(df.columns)}), tau_max={a.tau_max}, alpha={a.alpha}")
    res = run_pcmciplus(df, a.tau_max, a.alpha)
    links = extract_links(res, list(df.columns), a.tau_max)

    if links.empty:
        print("No significant links.")
    else:
        links = links.sort_values(["target", "lag", "source"]).reset_index(drop=True)
        print(links.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        links.to_csv(a.out, index=False)
        print(f"\nSaved {a.out}")

    if a.mode == "synthetic" and not a.shuffle:
        score_against_truth(links if not links.empty else
                            pd.DataFrame(columns=["source", "target", "lag"]),
                            SYN_NAMES)


if __name__ == "__main__":
    main()
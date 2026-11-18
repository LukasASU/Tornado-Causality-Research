#!/usr/bin/env python3
"""
STEP 3: causal discovery. Reads the preprocessed table and runs PCMCI+.

This script only does the causal analysis. All data loading and cleaning happens in
preprocess_data.py, which writes output/preprocessed_<outcome>.csv (standardized
monthly anomalies, one column per variable).

Run:
    python pcmci.py --outcome spc
    python pcmci.py --outcome both --tau-max 6
    python pcmci.py --outcome both --drop t2m favorable_days     # leave variables out

Writes to ./output:
    links_<outcome>.csv     table of significant links
    graph_<outcome>.png     causal graph with a legend
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


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
    "soil_moisture": "Soil Moisture\n(top layer)",
    "shear": "Deep-Layer\nWind Shear",
    "cape_hourly": "CAPE\n(hourly data)",
    "cape_x_shear": "CAPE × Shear",
    "favorable_days": "Favorable Storm\nDays per Month",
    "precip": "Total\nPrecipitation",
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
    "soil_moisture": "ERA5 volumetric soil water, layer 1, regional monthly mean",
    "shear": "wind speed difference, 500 hPa minus 10 m (m/s), regional mean",
    "cape_hourly": "ERA5 CAPE at the downloaded hours, regional mean",
    "cape_x_shear": "regional-mean CAPE times shear",
    "favorable_days": "log(1 + days per month with regional CAPE and shear both above thresholds)",
    "precip": "ERA5 total precipitation, regional monthly mean",
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
    "Orange glow + bold name = the outcome variable",
]
OUTCOMES = ("tornado_days", "sheldus_loss")


def label_of(name):
    return DISPLAY_NAMES[name] if name in DISPLAY_NAMES else name


def highlight_outcomes(ax, outcome_labels):
    """Add a soft orange glow behind the outcome node(s) and make their names bold.
    Finds each node by its label text, so it does not depend on tigramite internals;
    if a label is not found, that node is simply left unhighlighted."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for t in ax.texts:
        if t.get_text() in outcome_labels:
            x, y = t.get_position()
            ax.scatter([x], [y], s=16000, color="#f59e0b", alpha=0.35, zorder=0,
                       linewidths=0)
            t.set_fontweight("bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_graph_png(res, labels, raw_path, outcome_labels=()):
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
        try:
            highlight_outcomes(ax, set(outcome_labels))
        except Exception as e:   # the highlight is cosmetic; never lose the graph
            print(f"(could not highlight outcome: {e})")
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
        labels = [label_of(n) for n in names]
        outcome_labels = [label_of(n) for n in names if n in OUTCOMES]
        draw_graph_png(res, labels, raw, outcome_labels)
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
        lines = [label_of(n).replace("\n", " ") + ": "
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
OUTCOME_COLUMN = {"spc": "tornado_days", "sheldus": "sheldus_loss"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="output",
                    help="where preprocess_data.py wrote its files")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--outcome", choices=["spc", "sheldus", "both"], default="spc")
    ap.add_argument("--tau-max", type=int, default=3, help="longest lag to test, in months")
    ap.add_argument("--alpha", type=float, default=0.01, help="significance level")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="variables to leave out, e.g. --drop t2m")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    kinds = ["spc", "sheldus"] if a.outcome == "both" else [a.outcome]
    for kind in kinds:
        name = OUTCOME_COLUMN[kind]
        path = os.path.join(a.in_dir, f"preprocessed_{name}.csv")
        if not os.path.exists(path):
            sys.exit(f"{path} not found. Run preprocess_data.py --outcome {kind} first.")
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.drop(columns=a.drop, errors="ignore")
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        states = (json.load(open(meta_path)).get("states", [])
                  if os.path.exists(meta_path) else [])

        print(f"\n=== Run: outcome = {name} ===")
        print(f"PCMCI+ on {len(df)} months x {df.shape[1]} variables "
              f"(tau_max={a.tau_max}, alpha={a.alpha}): {', '.join(df.columns)}")
        res = run_pcmciplus(df, a.tau_max, a.alpha)
        links = extract_links(res, list(df.columns), a.tau_max)
        if links.empty:
            print("No significant links.")
        else:
            links = links.sort_values(["target", "lag", "source"]).reset_index(drop=True)
            print(links.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            links.to_csv(os.path.join(a.out_dir, f"links_{name}.csv"), index=False)
        save_graph(res, list(df.columns), os.path.join(a.out_dir, f"graph_{name}.png"),
                   states)
        print("(mark '-->' = directed; 'o-o' = same-month link whose direction is undetermined)")

    print(f"\nDone. Results in ./{a.out_dir}")


if __name__ == "__main__":
    main()
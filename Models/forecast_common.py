#!/usr/bin/env python3
"""
Shared code for the four forecast-model scripts (one per outcome variable):

    model_tornado_days.py   model_tornado_count.py   model_outbreak_days.py   model_sheldus_loss.py

Each script asks: do the causes that PCMCI found forecast the outcome better than simply
using every available variable?  Keep this file next to the four scripts.

What a script does
------------------
1. Reads, from ../PCMCI/output (relative to this folder):
       preprocessed_<outcome>.csv   the cleaned monthly anomalies (all variables)
       links_<outcome>.csv          the links PCMCI+ found
2. Takes the DIRECT PARENTS of the outcome with lag >= 1 month from the links file.
   Only these can forecast (a same-month link cannot be known ahead of time).
3. Tests several models on the same months, with an expanding-window split over time:
       train on early years, test on the next block of years, slide forward.
   Models (all ridge regression, so they are simple and fair):
       Climatology               predict the training-period average (anomaly ~ 0)
       Last month's value        predict the outcome's own previous month
       Own past only             ridge on the outcome's previous month
       Causal parents            ridge on the PCMCI parents (lagged)
       Causal parents + own past same, plus the outcome's previous month
       All features + own past   every variable at every lag up to max(3, longest parent lag)
       Same-month neighbors      NOT a forecast: variables linked in the same month; shows how
                                 much the current environment explains (an upper reference)
   Diagnostics: each parent alone, and the parent set with one parent removed.
4. Scores every model on the out-of-sample months:
       skill = 1 - (model squared error / climatology squared error)
       1 = perfect, 0 = no better than the average, negative = worse than the average
   with 95% confidence ranges from a moving-block bootstrap (blocks of consecutive months),
   for all months and for the tornado season (March-June) alone.
5. Writes tables and a chart to ./results (next to these scripts).

Important caveat: by default the parents come from links_<outcome>.csv, which PCMCI found
using ALL years, including the years used for testing. That is mildly optimistic for the
causal model. Use --reselect to re-run PCMCI+ inside every training window (slower,
needs tigramite) for a clean test.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_PCMCI_DIR = HERE.parent / "PCMCI"          # sibling folder of this one
SPRING = (3, 4, 5, 6)                              # tornado season months
CLIM = "Climatology (training average)"
PERSIST = "Last month's value"
OWN = "Own past only"
CAUSAL = "Causal parents"
CAUSAL_OWN = "Causal parents + own past"
ALLF = "All features + own past"
NOWCAST = "Same-month neighbors (nowcast, not a forecast)"
GROUP_ORDER = ["base", "causal", "all", "nowcast"]
GROUP_COLORS = {"base": "#8b8a85", "causal": "#2a78d6", "all": "#eb6834",
                "nowcast": "#1baf7a"}
GROUP_LABELS = {"base": "Baselines", "causal": "Causal parents (PCMCI)",
                "all": "All features", "nowcast": "Same-month (not a forecast)"}

Spec = Tuple[str, int]        # (variable name, lag in months)


# --------------------------------------------------------------------------
# Reading the PCMCI results
# --------------------------------------------------------------------------
def parents_from_links(links: pd.DataFrame, outcome: str) -> Tuple[List[Spec], List[str]]:
    """Lagged direct parents (lag >= 1) and same-month neighbors of the outcome.

    Same-month links are listed regardless of arrow direction, because PCMCI+ often points
    same-month arrows the wrong way (e.g. outcome -> CAPE)."""
    lagged: List[Spec] = []
    same: List[str] = []
    for r in links.to_dict("records"):
        src, tgt, lag = str(r["source"]), str(r["target"]), int(r["lag"])
        if lag >= 1 and tgt == outcome and src != outcome:
            lagged.append((src, lag))
        elif lag == 0:
            if tgt == outcome and src != outcome:
                same.append(src)
            elif src == outcome and tgt != outcome:
                same.append(tgt)
    return sorted(set(lagged)), sorted(set(same))


def load_pcmci_module(path: Path):
    """Import the PCMCI script by file path (its name has a hyphen, so a normal import fails)."""
    if not path.exists():
        sys.exit(f"PCMCI script not found at {path}. Use --pcmci-script.")
    spec = importlib.util.spec_from_file_location("pcmci_script", str(path))
    if spec is None or spec.loader is None:
        sys.exit(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reselect_links(mod, train_df: pd.DataFrame, tau_max: int, alpha: float) -> pd.DataFrame:
    """Run PCMCI+ on the training years only and return its links table."""
    res = mod.run_pcmciplus(train_df, tau_max, alpha)
    return mod.extract_links(res, list(train_df.columns), tau_max)


# --------------------------------------------------------------------------
# Learners (numpy only for ridge; scikit-learn only if --forest)
# --------------------------------------------------------------------------
ALPHAS = np.logspace(-1, 4, 26)


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    """Ridge regression; the penalty is chosen by leave-one-out error on the training rows.
    With no features it simply predicts the training average."""
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), float(ytr.mean()))
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = float(ytr.mean())
    yc = ytr - ym
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    Uty = U.T @ yc
    best_alpha, best_score = ALPHAS[0], np.inf
    for a in ALPHAS:
        shrink = S ** 2 / (S ** 2 + a)
        resid = yc - U @ (shrink * Uty)
        leverage = ((U ** 2) * shrink).sum(axis=1)
        score = float(np.mean((resid / (1.0 - leverage)) ** 2))
        if score < best_score:
            best_alpha, best_score = a, score
    coef = Vt.T @ ((S / (S ** 2 + best_alpha)) * Uty)
    return B @ coef + ym


def forest_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                       seed: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    if Xtr.shape[1] == 0:
        return np.full(len(Xte), float(ytr.mean()))
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                              random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------
def model_list(outcome: str, parents: List[Spec], same: List[str], predictors: List[str],
               all_lag: int, with_forest: bool, diagnostics: bool):
    """Each model: (name, group, learner, feature list). learner: ridge/forest/persist."""
    own: List[Spec] = [(outcome, 1)]
    all_specs: List[Spec] = [(v, k) for v in predictors for k in range(1, all_lag + 1)] + own
    models: List[Tuple[str, str, str, List[Spec]]] = [
        (CLIM, "base", "ridge", []),
        (PERSIST, "base", "persist", []),
        (OWN, "base", "ridge", own),
    ]
    if parents:
        models += [(CAUSAL, "causal", "ridge", list(parents)),
                   (CAUSAL_OWN, "causal", "ridge", list(parents) + own)]
    models.append((ALLF, "all", "ridge", all_specs))
    if with_forest:
        if parents:
            models.append((CAUSAL_OWN + " (forest)", "causal", "forest",
                           list(parents) + own))
        models.append((ALLF + " (forest)", "all", "forest", all_specs))
    if same:
        models.append((NOWCAST, "nowcast", "ridge", [(v, 0) for v in same]))
    if diagnostics and parents:
        for p in parents:
            models.append((f"Only {p[0]}(t-{p[1]})", "diag", "ridge", [p]))
        if len(parents) >= 2:
            for p in parents:
                rest = [q for q in parents if q != p]
                models.append((f"Parents without {p[0]}(t-{p[1]})", "diag", "ridge", rest))
    return models


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_table(y: np.ndarray, preds: Dict[str, np.ndarray], mask: np.ndarray,
                groups: Dict[str, str], block: int, n_boot: int, seed: int) -> pd.DataFrame:
    """Skill vs climatology with moving-block-bootstrap 95% ranges, plus RMSE, correlation,
    and the skill difference to the all-features model."""
    sel = np.where(mask)[0]
    n = len(sel)
    if n < 12:
        return pd.DataFrame()
    obs = y[sel]
    names = list(preds)
    sq = {m: (obs - preds[m][sel]) ** 2 for m in names}
    base = sq[CLIM]
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 0) + 1, size=(n_boot, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    idx = np.minimum(idx, n - 1)
    base_boot = base[idx].sum(axis=1)
    boot = {m: 1.0 - sq[m][idx].sum(axis=1) / base_boot for m in names}
    ref = ALLF if ALLF in boot else None
    rows = []
    for m in names:
        skill = 1.0 - sq[m].sum() / base.sum()
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        p = preds[m][sel]
        corr = float(np.corrcoef(p, obs)[0, 1]) if (m != CLIM and np.std(p) > 0) else np.nan
        row = {"model": m, "group": groups[m], "skill": skill, "skill_lo": lo,
               "skill_hi": hi, "rmse": float(np.sqrt(sq[m].mean())), "corr": corr}
        if ref and m != ref:
            d = boot[m] - boot[ref]
            row["diff_vs_all"] = skill - (1.0 - sq[ref].sum() / base.sum())
            row["diff_lo"], row["diff_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, label: str) -> None:
    """A few plain-language lines about the comparison that matters."""
    if table.empty:
        return
    t = table.set_index("model")
    print(f"\n  Reading the {label} results:")
    for m in (CAUSAL_OWN, CAUSAL, ALLF, NOWCAST):
        if m not in t.index:
            continue
        r = t.loc[m]
        if r["skill_lo"] > 0:
            word = "beats the climatology baseline"
        elif r["skill_hi"] < 0:
            word = "is WORSE than the climatology baseline"
        else:
            word = "cannot be told apart from the climatology baseline"
        print(f"   - {m}: skill {r['skill']:+.3f} [{r['skill_lo']:+.3f}, "
              f"{r['skill_hi']:+.3f}] -> {word}")
    if CAUSAL_OWN in t.index and "diff_lo" in t.columns:
        r = t.loc[CAUSAL_OWN]
        if r["diff_lo"] > 0:
            word = "the causal parents forecast BETTER than all features"
        elif r["diff_hi"] < 0:
            word = "the causal parents forecast WORSE than all features"
        else:
            word = "no clear difference between causal parents and all features"
        print(f"   - Causal parents + own past minus all features: "
              f"{r['diff_vs_all']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] -> {word}")


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def skill_chart(tables: Dict[str, pd.DataFrame], outcome: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    shown = {k: t[t["group"] != "diag"] for k, t in tables.items() if not t.empty}
    if not shown:
        return
    first = next(iter(shown.values()))
    order = first["model"].tolist()[::-1]                      # first model on top
    fig, axes = plt.subplots(1, len(shown), figsize=(6.4 * len(shown), 0.55 * len(order) + 2),
                             sharey=True, facecolor=surface)
    for ax, (title, t) in zip(np.atleast_1d(axes), shown.items()):
        t = t.set_index("model").reindex(order)
        ypos = np.arange(len(order))
        ax.set_facecolor(surface)
        ax.barh(ypos, t["skill"], height=0.5,
                color=[GROUP_COLORS[g] for g in t["group"]], zorder=3)
        ax.errorbar(t["skill"], ypos,
                    xerr=[t["skill"] - t["skill_lo"], t["skill_hi"] - t["skill"]],
                    fmt="none", ecolor=ink2, elinewidth=1.2, capsize=3, zorder=4)
        ax.axvline(0, color=ink2, lw=1.2, zorder=5)
        ax.grid(axis="x", color="#e7e6e2", lw=0.8, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(order, fontsize=9, color=ink)
        ax.tick_params(axis="x", colors=ink2, labelsize=9)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=ink)
        ax.set_xlabel("Skill vs climatology  (0 = no better than average, <0 = worse)",
                      fontsize=9, color=ink2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#cfcec9")
        # "last month's value" is often hugely negative; zoom on the other models
        core = t[t.index != PERSIST]
        x_lo = min(-0.1, float(core["skill_lo"].min()) - 0.05)
        x_hi = max(0.1, float(core["skill_hi"].max())) + 0.12
        ax.set_xlim(x_lo, x_hi)
        for yv, (_, r) in zip(ypos, t.iterrows()):
            if r["skill"] < x_lo:
                ax.text(x_lo + 0.01, yv, f"off scale: {r['skill']:+.2f}", va="center",
                        fontsize=8, color=ink, zorder=6,
                        bbox=dict(fc=surface, ec="none", pad=1.5))
            else:
                ax.text(max(r["skill_hi"], 0) + 0.004, yv, f"{r['skill']:+.3f}",
                        va="center", fontsize=8, color=ink2)
    handles = [Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Out-of-sample forecast skill: {outcome}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(outcome: str) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcmci-dir", default=str(DEFAULT_PCMCI_DIR),
                    help="the PCMCI project folder (default: ../PCMCI)")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--first-test-year", type=int, default=2002,
                    help="first year that is ever used for testing (earlier years only train)")
    ap.add_argument("--block-years", type=int, default=5,
                    help="length of each test block in years")
    ap.add_argument("--all-max-lag", type=int, default=3,
                    help="all-features model uses lags 1..this (raised to the longest parent lag)")
    ap.add_argument("--drop", nargs="*", default=[],
                    help="variables to leave out of everything, e.g. --drop t2m")
    ap.add_argument("--forest", action="store_true",
                    help="also test random forests (needs scikit-learn)")
    ap.add_argument("--reselect", action="store_true",
                    help="re-run PCMCI+ on each training window to choose parents "
                         "(clean test; slower; needs tigramite)")
    ap.add_argument("--tau-max", type=int, default=6, help="with --reselect")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="with --reselect; use the same value as your PCMCI run")
    ap.add_argument("--pcmci-script", default=None,
                    help="path to run-PCMCI.py (default: <pcmci-dir>/run-PCMCI.py)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pcmci_dir = Path(a.pcmci_dir)
    out_dir = pcmci_dir / "output"
    pre_path = out_dir / f"preprocessed_{outcome}.csv"
    links_path = out_dir / f"links_{outcome}.csv"
    for p in (pre_path, links_path):
        if not p.exists():
            sys.exit(f"{p} not found. Run PreProcessData.py and run-PCMCI.py first "
                     f"(looking in {out_dir}).")
    results = Path(a.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pre_path, parse_dates=["date"], index_col="date")
    df = df.drop(columns=a.drop, errors="ignore")
    links = pd.read_csv(links_path)
    predictors = [c for c in df.columns if c != outcome]
    dates = pd.DatetimeIndex(df.index)
    years = np.asarray(dates.year)

    # ---- folds: expanding window, each test block follows its training years
    last_year = int(years.max())
    folds = []
    y0 = a.first_test_year
    while y0 <= last_year:
        folds.append((y0, min(y0 + a.block_years - 1, last_year)))
        y0 += a.block_years
    if not folds:
        sys.exit("No test years: lower --first-test-year.")

    # ---- parents (one global set, or one per fold with --reselect)
    mod = None
    if a.reselect:
        mod = load_pcmci_module(Path(a.pcmci_script) if a.pcmci_script
                                else pcmci_dir / "run-PCMCI.py")
    fold_parents: List[List[Spec]] = []
    fold_same: List[List[str]] = []
    for (ys, ye) in folds:
        if mod is not None:
            train = df[years < ys]
            print(f"Selecting parents on {int(years.min())}-{ys - 1} "
                  f"(test block {ys}-{ye}) ...")
            fl = reselect_links(mod, train, a.tau_max, a.alpha)
        else:
            fl = links
        par, same = parents_from_links(fl, outcome)
        par = [p for p in par if p[0] in df.columns]
        same = [s for s in same if s in df.columns]
        fold_parents.append(par)
        fold_same.append(same)

    global_parents = sorted({p for fp in fold_parents for p in fp})
    print(f"\n=== {outcome} ===")
    if mod is None:
        print(f"Lagged parents in {links_path.name}: "
              f"{', '.join(f'{v}(t-{k})' for v, k in global_parents) or 'NONE'}")
        print(f"Same-month neighbors: {', '.join(fold_same[0]) or 'none'}")
        if not global_parents:
            print("\nNo variable with lag >= 1 points at this outcome, so there is nothing "
                  "to forecast it with.\nThe causal-parent models are skipped. Baselines, "
                  "all features and the same-month (nowcast) model are still scored for "
                  "comparison.")
    else:
        for (ys, ye), par in zip(folds, fold_parents):
            print(f"  block {ys}-{ye}: "
                  f"{', '.join(f'{v}(t-{k})' for v, k in par) or 'no lagged parents'}")

    longest = max([k for fp in fold_parents for _, k in fp] + [0])
    all_lag = max(a.all_max_lag, longest, 1)
    skip = max(all_lag, a.tau_max if a.reselect else 0, 1)   # rows lost to lagging
    valid = np.arange(len(df)) >= skip
    y = df[outcome].to_numpy(dtype=float)

    cache: Dict[Tuple[Spec, ...], np.ndarray] = {}

    def design(specs: List[Spec]) -> np.ndarray:
        key = tuple(specs)
        if key not in cache:
            cols = [df[v].shift(k).to_numpy(dtype=float) for v, k in specs]
            cache[key] = np.column_stack(cols) if cols else np.empty((len(df), 0))
        return cache[key]

    persist = df[outcome].shift(1).to_numpy(dtype=float)

    # ---- fit and predict, fold by fold
    preds: Dict[str, np.ndarray] = {}
    groups: Dict[str, str] = {}
    tested = np.zeros(len(df), dtype=bool)
    include_diag = not a.reselect
    for (ys, ye), par, same in zip(folds, fold_parents, fold_same):
        train_rows = valid & (years < ys)
        test_rows = valid & (years >= ys) & (years <= ye)
        if train_rows.sum() < 60 or not test_rows.any():
            print(f"  (skipping block {ys}-{ye}: too little training data)")
            continue
        tested |= test_rows
        # with --reselect a fold may have no parents: models fall back to their baselines
        specs_par = par if (par or not a.reselect) else []
        for name, grp, learner, specs in model_list(outcome, specs_par, same, predictors,
                                                    all_lag, a.forest, include_diag):
            if name not in preds:
                preds[name] = np.full(len(df), np.nan)
                groups[name] = grp
            if learner == "persist":
                preds[name][test_rows] = persist[test_rows]
                continue
            X = design(specs)
            if learner == "forest":
                out = forest_fit_predict(X[train_rows], y[train_rows], X[test_rows], a.seed)
            else:
                out = ridge_fit_predict(X[train_rows], y[train_rows], X[test_rows])
            preds[name][test_rows] = out

    if not tested.any():
        sys.exit("No test months. Check --first-test-year / the data range.")
    # a model that only exists for some folds (reselect) cannot be scored fairly
    preds = {m: p for m, p in preds.items() if not np.isnan(p[tested]).any()}
    groups = {m: groups[m] for m in preds}
    if CAUSAL not in preds and CAUSAL_OWN not in preds and a.reselect:
        print("PCMCI+ found no lagged parents in any training window.")

    month = np.asarray(dates.month)
    block_all = 12
    main_m = {m: p for m, p in preds.items() if groups[m] != "diag"}
    diag_m = {m: p for m, p in preds.items() if groups[m] == "diag"}

    tables: Dict[str, pd.DataFrame] = {}
    for label, mask, block in (("All months", tested, block_all),
                               ("Tornado season (Mar-Jun)",
                                tested & np.isin(month, SPRING), 4)):
        tab = score_table(y, preds, mask, groups, block, a.boot, a.seed)
        tables[label] = tab
        if tab.empty:
            continue
        print(f"\n--- {label}: {int(mask.sum())} test months, "
              f"{int(years[mask].min())}-{int(years[mask].max())} ---")
        show = tab[tab["group"] != "diag"].copy()
        show["order"] = show["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
        show = show.sort_values("order", kind="stable").drop(columns=["order", "group"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        verdict(tab, label.lower())
        dtab = tab[tab["group"] == "diag"].drop(columns=["group"])
        if not dtab.empty:
            print(f"\n  Diagnostics, {label.lower()} (each parent alone / one removed):")
            print(dtab[["model", "skill", "skill_lo", "skill_hi"]].to_string(
                index=False, float_format=lambda v: f"{v:+.3f}"))
        tab.to_csv(results / f"{outcome}_skill_{'spring' if 'season' in label else 'all'}.csv",
                   index=False)

    pd.DataFrame(preds, index=df.index).assign(observed=y)[tested].to_csv(
        results / f"{outcome}_predictions.csv")
    pd.DataFrame([{"test_block": f"{ys}-{ye}",
                   "lagged_parents": "; ".join(f"{v}(t-{k})" for v, k in par),
                   "same_month_neighbors": "; ".join(same)}
                  for (ys, ye), par, same in zip(folds, fold_parents, fold_same)]
                 ).to_csv(results / f"{outcome}_parents_by_fold.csv", index=False)
    skill_chart(tables, outcome, results / f"{outcome}_skill.png")
    print(f"\nWrote tables and chart to {results}")

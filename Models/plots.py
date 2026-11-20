#!/usr/bin/env python3
"""
The observed-vs-predicted chart for the forecast scripts (used by forecast_common.py).

Colors follow the model: grey = baseline, blue = causal parents (PCMCI),
orange = all features, green = same-month nowcast (not a forecast).
"""
from pathlib import Path
from textwrap import fill
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE, GRID, OBS = "#0b0b0b", "#52514e", "#fcfcfb", "#e7e6e2", "#c9c8c2"
COLORS = {"base": "#8b8a85", "causal": "#2a78d6", "all": "#eb6834", "nowcast": "#1baf7a"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def pick_models(preds: Dict[str, np.ndarray], baseline: str, causal: Sequence[str],
                all_name: str, nowcast: str) -> List[str]:
    """The models worth drawing: baseline, one causal model, all features, nowcast."""
    chosen = [baseline]
    for c in causal:                       # first causal model that exists
        if c in preds:
            chosen.append(c)
            break
    chosen += [m for m in (all_name, nowcast) if m in preds]
    return [m for m in chosen if m in preds]


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcec9")
    ax.tick_params(colors=INK2, labelsize=9)


def _figure(w, h, title, subtitle=None):
    """Figure with a title and subtitle placed in absolute inches from the top."""
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / h, ha="left", va="top", fontsize=13,
                 fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.45 / h, subtitle, ha="left", va="top", fontsize=9, color=INK2)
    return fig


def _save(fig, path: Path):
    header = 0.85 / fig.get_figheight()      # share of the height kept for the heading
    fig.tight_layout(rect=(0, 0, 1, 1 - header))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _shade_spring(ax, dates: pd.DatetimeIndex):
    for y in sorted(set(int(v) for v in dates.year)):
        ax.axvspan(pd.Timestamp(y, 3, 1), pd.Timestamp(y, 7, 1), color="#f3e7c9", alpha=0.45,
                   lw=0, zorder=0)


def timeline_regression(path: Path, dates: pd.DatetimeIndex, obs: np.ndarray,
                        preds: Dict[str, np.ndarray], models: List[str],
                        groups: Dict[str, str], skills: Dict[str, float],
                        outcome: str) -> None:
    """Observed vs predicted month by month, one panel per model."""
    fig = _figure(14, 2.1 * len(models) + 1.2, f"Observed vs predicted, {outcome}",
                  "Grey = observed anomaly; colored line = the model's forecast; "
                  "shaded bands = March-June")
    axes = fig.subplots(len(models), 1, sharex=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        _style(ax)
        _shade_spring(ax, dates)
        ax.plot(dates, obs, color=OBS, lw=1.0, zorder=2)
        ax.plot(dates, preds[m], color=COLORS[groups[m]], lw=1.8, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_ylabel(fill(m, 22), fontsize=8, color=INK, rotation=0, ha="right",
                      va="center", labelpad=8)
        if m in skills:
            ax.text(0.995, 0.88, f"skill {skills[m]:+.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=INK2)
    np.atleast_1d(axes)[-1].set_xlabel("Month", color=INK2, fontsize=9)
    _save(fig, path)
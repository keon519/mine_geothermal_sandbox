"""Figures from RunResult objects. Grid-agnostic: every spatial view comes
from RunResult.along_drift / wall_profile / section."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .results import RunResult

# fixed categorical order (never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEAT = LinearSegmentedColormap.from_list("heat1", ["#fdf3ea", "#f5b27a", "#d95926", "#7a2a0c"])
FLUSH_DAYS = 5.0   # ignore the initial cold-water flush when scaling heat axes

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "lines.linewidth": 2.0, "legend.frameon": False,
})


def _heat_ylim(ax, results):
    """Scale a heat axis to the steady part of the run.

    The first few days show a huge spike while cold water that filled the workings is flushed
    out, which would squash everything after it. This sets the axis from what comes after that
    flush.
    """
    tops = []
    for r in results:
        late = r.heat_removed_W[r.times_d > min(FLUSH_DAYS, 0.5 * r.times_d[-1])]
        if late.size:
            tops.append(late.max() / 1e6)
    if tops:
        ax.set_ylim(0, max(1.3 * max(tops), 1e-3))


def _save(fig, path):
    """Tidy the layout and save the figure if a path was given; returns the figure either way."""
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_run(res: RunResult, title: str = "", path: str | None = None):
    """Loop temperatures, heat, T along the drift, rock profile."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    t = res.times_d

    ax = axs[0, 0]
    ax.plot(t, res.T_in_C, color=SERIES[1], label="T_in (Pump In)")
    ax.plot(t, res.T_out_C, color=SERIES[0], label="T_out (Pump Out)")
    ax.set(xlabel="time [d]", ylabel="water temperature [°C]", title="Loop temperatures")
    ax.legend(loc="lower right")

    ax = axs[0, 1]
    ax.plot(t, res.heat_removed_W / 1e6, color=SERIES[0], label="removed by the mine")
    if res.Q_load_W is not None and np.all(np.isfinite(res.Q_load_W)):
        ax.plot(t, res.Q_load_W / 1e6, color=SERIES[1], ls="--", label="surface heat load")
        ax.legend(loc="upper right")
    ax.set(xlabel="time [d]", ylabel="heat [MW]", title="Heat rejected into the rock")
    _heat_ylim(ax, [res])

    ax = axs[1, 0]
    x, T = res.along_drift()
    ax.plot(x, T, color=SERIES[0])
    ax.set(xlabel="distance along drift from Pump In [m]", ylabel="water temperature [°C]",
           title=f"Along the drift at t = {t[-1]:.0f} d")

    ax = axs[1, 1]
    d, T = res.wall_profile()
    ax.plot(d, T, color=SERIES[1], marker="o", ms=4)
    ax.set(xlabel=res.graph.cross_axis_label, ylabel="rock temperature [°C]",
           title=f"Rock above mid-drift at t = {t[-1]:.0f} d", xscale="log")
    if title:
        fig.suptitle(title, fontsize=12)
    return _save(fig, path)


def plot_sweep(results: dict, key_label: str, quantity: str = "T_out", title: str = "",
               path: str | None = None):
    """Overlay several runs. quantity: 'T_out', 'heat' or 'effectiveness'."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    pick = {"T_out": (lambda r: r.T_out_C, "T_out [°C]"),
            "heat": (lambda r: r.heat_removed_W / 1e6, "heat removed by the mine [MW]"),
            "effectiveness": (lambda r: 100 * r.effectiveness, "cooling achieved [% of possible]")}
    f, ylabel = pick[quantity]
    for k, (label, res) in enumerate(results.items()):
        ax.plot(res.times_d, f(res), color=SERIES[k % len(SERIES)], label=str(label))
    ax.set(xlabel="time [d]", ylabel=ylabel, title=title or f"Sweep over {key_label}")
    if quantity == "heat":
        _heat_ylim(ax, results.values())
    if quantity == "effectiveness":
        ax.set_ylim(0, None)
    ax.legend(title=key_label, loc="best")
    return _save(fig, path)


def plot_section(res: RunResult, step: int = -1, title: str = "", path: str | None = None):
    """Temperature in the vertical plane through the drift."""
    xs, zs, img = res.section(step)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pm = ax.pcolormesh(xs, zs, img, cmap=HEAT, shading="nearest")
    nz = np.abs(zs[zs != 0])
    half = float(nz.min()) if nz.size else 1.0
    ax.set(xlabel="distance along drift [m]", ylabel=res.graph.cross_axis_label,
           title=title or f"Temperature at t = {res.times_d[step]:.0f} d")
    ax.set_yscale("symlog", linthresh=max(half, 0.5))
    ax.grid(False)
    fig.colorbar(pm, ax=ax, label="°C")
    return _save(fig, path)

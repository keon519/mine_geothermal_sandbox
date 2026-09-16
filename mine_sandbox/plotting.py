"""Matplotlib helpers for the sandbox results."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# fixed categorical order (never cycled); a 9th series folds into "other"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HEAT = LinearSegmentedColormap.from_list("heat1", ["#fdf3ea", "#f5b27a", "#d95926", "#7a2a0c"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "lines.linewidth": 2.0, "legend.frameon": False,
})


def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK2)


def plot_single_run(res, title="", path=None):
    """T_in / T_out vs time, heat removed vs time, T along shaft, radial rock profile."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    t = res.times_d
    ax = axs[0, 0]
    ax.plot(t, res.T_in_C, color=SERIES[1], label="T_in (Pump In)")
    ax.plot(t, res.T_out_C, color=SERIES[0], label="T_out (Pump Out)")
    ax.set(xlabel="time [d]", ylabel="water temperature [°C]", title="Loop temperatures")
    ax.legend(loc="lower right")

    ax = axs[0, 1]
    ax.plot(t, res.heat_removed_W / 1e6, color=SERIES[0], label="removed by the mine")
    if hasattr(res, "Q_load_W"):
        ax.plot(t, np.asarray(res.Q_load_W) / 1e6, color=SERIES[1], ls="--", label="SMR heat load")
        ax.legend(loc="upper right")
    ax.set(xlabel="time [d]", ylabel="heat [MW]", title="Heat rejected into the rock")
    # the first day is dominated by flushing the cold water that filled the shaft;
    # don't let that spike squash the year-long signal
    later = res.heat_removed_W[t > min(5.0, 0.5 * t[-1])] / 1e6
    if later.size:
        ax.set_ylim(0, max(1.5 * later.max(), 1e-3))

    ax = axs[1, 0]
    ax.plot(res.x_m, res.T_shaft_final, color=SERIES[0])
    ax.set(xlabel="distance along shaft from Pump In [m]", ylabel="water temperature [°C]",
           title=f"Temperature along the shaft at t = {t[-1]:.0f} d")

    ax = axs[1, 1]
    mid = res.temp_final.shape[0] // 2
    ax.plot(res.r_m[1:], res.temp_final[mid, 1:], color=SERIES[1], marker="o", ms=4)
    ax.set(xlabel="radius from shaft axis [m]", ylabel="rock temperature [°C]",
           title=f"Rock temperature profile (mid-shaft) at t = {t[-1]:.0f} d", xscale="log")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_sweep(results: dict, key_label: str, title="", path=None, quantity="T_out"):
    """Overlay T_out(t) (or heat removed) for several runs. `results` maps label -> result."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, (label, res) in enumerate(results.items()):
        c = SERIES[k % len(SERIES)]
        if quantity == "T_out":
            y = res.T_out_C; ylab = "T_out [°C]"
        else:
            y = res.heat_removed_W / 1e6; ylab = "heat removed by the mine [MW]"
        ax.plot(res.times_d, y, color=c, label=str(label))
    ax.set(xlabel="time [d]", ylabel=ylab, title=title or f"Sweep over {key_label}")
    if quantity != "T_out":
        # skip the first-day flushing spike when choosing the y-range
        tops = [np.max(r.heat_removed_W[r.times_d > min(5.0, 0.5 * r.times_d[-1])]) / 1e6
                for r in results.values()]
        ax.set_ylim(0, 1.3 * max(tops))
    ax.legend(title=key_label, loc="best")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_field(res, path=None, title=""):
    """x–r temperature map of the shaft and rock at the end of the run."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    T = res.temp_final.T                     # (1+n_r, n_x)
    x = res.x_m
    r = np.concatenate([[0.0], res.r_m[1:]])
    pm = ax.pcolormesh(x, r, T, cmap=HEAT, shading="nearest")
    ax.set(xlabel="distance along shaft [m]", ylabel="radius from shaft axis [m]", yscale="symlog",
           title=title or f"Temperature field at t = {res.times_d[-1]:.0f} d")
    ax.grid(False)
    fig.colorbar(pm, ax=ax, label="°C")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

"""Mine-water heat-sink sandbox - example driver.

    python run_sandbox.py                  # radial grid: baseline, 4 sweeps, coupled runs
    python run_sandbox.py --grid 3d        # same study on the 3-D block grid
    python run_sandbox.py --grid 2d --quick
    python run_sandbox.py --layered        # 3-D, drift in a coal seam under shale/sandstone

Edit BASE and SWEEPS below to change the study. Figures and a CSV land in
figures/<grid>/, MODFLOW files in mf6_run/.
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt

from mine_sandbox import (ROCKS, FixedHeatLoad, Layer, LayeredGeology, RadialDriftGrid,
                          StructuredDriftGrid, TespySurfaceLoop, make_scenario, run_coupled,
                          run_fixed_inlet)
from mine_sandbox.plotting import plot_run, plot_section, plot_sweep

GRIDS = {"radial": RadialDriftGrid(), "2d": StructuredDriftGrid.plane_2d(),
         "3d": StructuredDriftGrid.block_3d()}

# Baseline: 100 L/s of 35 C water through a 500 m, 4 m drift in 13 C sandstone
BASE = dict(flow_m3_h=360.0, T_in_C=35.0, length_m=500.0, diameter_m=4.0, rock="sandstone",
            T_rock_C=13.0, sim_days=365.0, dt_days=0.5)

# The four study variables: keyword -> (values, axis label)
SWEEPS = {
    "flow_m3_h":  ([90.0, 180.0, 360.0, 720.0], "flow [m³/h]"),
    "T_in_C":     ([25.0, 35.0, 45.0, 60.0], "T_in [°C]"),
    "length_m":   ([250.0, 500.0, 1000.0, 2000.0], "drift length [m]"),
    "diameter_m": ([2.0, 3.0, 4.0, 6.0], "drift diameter [m]"),
    "rock":       (["coal", "shale", "sandstone", "limestone", "granite"], "rock"),
}

COAL_SEAM = LayeredGeology(background=ROCKS["shale"],
                           layers=(Layer(-1.0, 1.0, ROCKS["coal"]),
                                   Layer(4.0, 1e9, ROCKS["sandstone"])))


def main():
    """Run the single-drift study: a baseline, a sweep over each study variable, then closed-loop
    runs.

    The baseline holds the inlet temperature fixed and plots the result plus a vertical section.
    Each sweep varies one knob from BASE (flow, inlet temperature, length, size, rock) and
    records conditions at day 30 and at the end in a CSV. The coupled runs close the loop
    through TESPy with a fixed surface heat load, which shows how hot the loop settles and
    whether the drift can keep up.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=GRIDS, default="radial")
    ap.add_argument("--quick", action="store_true", help="90-day runs with 1-day steps")
    ap.add_argument("--no-coupled", action="store_true")
    ap.add_argument("--layered", action="store_true", help="coal-seam geology (needs 2d/3d grid)")
    args = ap.parse_args()
    if args.layered and args.grid == "radial":
        args.grid = "3d"

    base = dict(BASE, grid=GRIDS[args.grid], workspace=os.path.join("mf6_run", args.grid))
    if args.quick:
        base.update(sim_days=90.0, dt_days=1.0)
    if args.layered:
        base.update(rock=COAL_SEAM, width_m=6.0, height_m=2.0, diameter_m=None)
    out = os.path.join("figures", args.grid + ("_layered" if args.layered else ""))
    os.makedirs(out, exist_ok=True)

    # 1. baseline, fixed inlet temperature
    t0 = time.time()
    sc = make_scenario(name="baseline", **base)
    res = run_fixed_inlet(sc)
    print(res.summary(), f"  ({time.time() - t0:.1f} s)")
    plot_run(res, title="Baseline - fixed inlet temperature", path=f"{out}/baseline.png")
    plot_section(res, path=f"{out}/baseline_section.png")
    plt.close("all")

    # 2. the four sweeps
    rows = []
    for key, (values, label) in SWEEPS.items():
        if args.layered and key in ("rock", "diameter_m"):
            continue
        results = {}
        for v in values:
            r = run_fixed_inlet(make_scenario(**{**base, key: v}, name=f"{key}_{v}"))
            results[v] = r
            i30 = r.at_day(min(30.0, r.times_d[-1]))
            rows.append((key, v, r.T_out_C[i30], r.heat_removed_W[i30] / 1e6, 100 * r.effectiveness[i30],
                         r.T_out_C[-1], r.heat_removed_W[-1] / 1e6, 100 * r.effectiveness[-1]))
            print(f"[{key:10s}] {v!s:10s} day30: T_out={rows[-1][2]:6.2f} C  "
                  f"{rows[-1][3]:6.3f} MW  eff={rows[-1][4]:5.1f}%")
        for q in ("T_out", "heat", "effectiveness"):
            plot_sweep(results, label, quantity=q, path=f"{out}/sweep_{key}_{q}.png")
        plt.close("all")
    with open(f"{out}/sweep_summary.csv", "w") as f:
        f.write("variable,value,T_out_day30_C,heat_day30_MW,effectiveness_day30_pct,"
                "T_out_end_C,heat_end_MW,effectiveness_end_pct\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # 3. closed loop with a fixed surface heat load (TESPy)
    if not args.no_coupled:
        for Q_MW in (0.1, 0.3, 1.0):
            sc = make_scenario(name=f"coupled_{Q_MW}MW", **base)
            t0 = time.time()
            r = run_coupled(sc, TespySurfaceLoop.for_scenario(sc, FixedHeatLoad(Q_MW * 1e6)))
            print(r.summary(), f"  ({time.time() - t0:.1f} s)")
            plot_run(r, title=f"Coupled loop, surface load {Q_MW} MW", path=f"{out}/coupled_{Q_MW}MW.png")
            plt.close("all")
    print(f"\nFigures in ./{out}/")


if __name__ == "__main__":
    main()

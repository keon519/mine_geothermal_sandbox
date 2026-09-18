
"""
Everything is driven by SandboxConfig (mine_sandbox/config.py). The four
variables are flow_m3_h, T_in_C, shaft_length_m /
shaft_diameter_m, and rock. Edit the SWEEPS dict below to change the values.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

FIG ="figures"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mine_sandbox import SandboxConfig, MineshaftModel, run_coupled, ROCKS  # noqa: E402
from mine_sandbox.plotting import plot_single_run, plot_sweep, plot_field, plt  # noqa: E402


# Baseline: 100 L/s of 35 °C water through a 500 m, 4 m drift in sandstone
BASE = SandboxConfig(
    flow_m3_h=360.0, T_in_C=35.0, shaft_length_m=500.0, shaft_diameter_m=4.0,
    rock="sandstone", T_rock_C=13.0, sim_days=365.0, dt_days=0.5,
    workspace="mf6_run",
)


SWEEPS = {
    "flow":     ("flow_m3_h",       [90.0, 180.0, 360.0, 720.0],          "flow [m³/h]"),
    "T_in":     ("T_in_C",          [25.0, 35.0, 45.0, 60.0],             "T_in [°C]"),
    "length":   ("shaft_length_m",  [250.0, 500.0, 1000.0, 2000.0],       "shaft length [m]"),
    "diameter": ("shaft_diameter_m", [2.0, 3.0, 4.0, 6.0],                "shaft diameter [m]"),
    "rock":     ("rock",            ["coal", "shale", "sandstone", "limestone", "granite"], "rock"),
}


def run_fixed_inlet(cfg: SandboxConfig):
    """MODFLOW only: constant T_in and flow, see what comes out."""
    return MineshaftModel(cfg).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="90-day runs, coarser steps")
    ap.add_argument("--no-coupled", action="store_true", help="skip the TESPy-coupled example")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    base = replace(BASE, sim_days=90.0, dt_days=1.0) if args.quick else BASE

    # baseline, fixed inlet temperature
    t0 = time.time()
    res = run_fixed_inlet(base)
    print(f"[baseline] {base.summary()}")
    print(f"           T_out at end = {res.T_out_C[-1]:.2f} C, "
          f"heat removed = {res.heat_removed_W[-1]/1e6:.3f} MW  ({time.time()-t0:.1f} s)")
    plot_single_run(res, title="Baseline — fixed inlet temperature", path=f"{FIG}/baseline.png")
    plot_field(res, path=f"{FIG}/baseline_field.png")
    plt.close("all")

    # the four sweeps
    table = []
    for name, (field, values, label) in SWEEPS.items():
        results = {}
        for v in values:
            cfg = replace(base, **{field: v}, name=f"{name}_{str(v).replace('.', 'p')}")
            r = run_fixed_inlet(cfg)
            results[v] = r
            table.append((name, v, r.T_out_C[-1], r.heat_removed_W[-1] / 1e6,
                          MineshaftModel(cfg).rock_heat_stored_J(r.temp_final) / 1e12))
            print(f"[{name:8s}] {field}={v!s:10s} T_out={r.T_out_C[-1]:6.2f} C  "
                  f"heat={r.heat_removed_W[-1]/1e6:6.3f} MW")
        plot_sweep(results, label, title=f"Effect of {label} on outlet temperature",
                   path=f"{FIG}/sweep_{name}_Tout.png")
        plot_sweep(results, label, title=f"Effect of {label} on heat rejected",
                   path=f"{FIG}/sweep_{name}_heat.png", quantity="heat")
        plt.close("all")

    with open(f"{FIG}/sweep_summary.csv", "w") as f:
        f.write("sweep,value,T_out_final_C,heat_removed_final_MW,rock_heat_stored_TJ\n")
        for row in table:
            f.write(",".join(str(x) for x in row) + "\n")

    # TESPy closed loop with a fixed SMR heat load
    if not args.no_coupled:
        for Q_MW in (0.1, 0.3, 1.0):
            cfg = replace(base, Q_load_W=Q_MW * 1e6, name=f"coupled_{Q_MW}MW")
            t0 = time.time()
            cr = run_coupled(cfg)
            print(f"[coupled ] Q_load={Q_MW} MW  ({time.time()-t0:.1f} s)\n{cr.summary()}")
            plot_single_run(cr, title=f"TESPy-coupled loop, SMR load {Q_MW} MW",
                            path=f"{FIG}/coupled_{Q_MW}MW.png")
            plt.close("all")

    print(f"\nFigures written to ./{FIG}/")


if __name__ == "__main__":
    main()

"""Run the same test on every mine file in a folder and compare them.

    python run_mines.py examples/mines                      # fixed 35 °C inlet, 365 days, fast engine
    python run_mines.py mines/ --engine mf6 --days 180
    python run_mines.py mines/ --load-mw 0.5               # closed loop with a TESPy heat load
    python run_mines.py mines/ --jobs 4                    # four mines at a time, one process each
    python run_mines.py mines/ --device auto               # GPU if one works (fv engine)

Writes figures/mines/comparison.csv and comparison.png: heat removed,
cooling effectiveness and heat per square metre of wall for each mine.
Generate test mines with:  python -m mine_sandbox.mines --count 10 --size medium --out mines/
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from mine_sandbox import ConstantInlet, FixedHeatLoad, TespySurfaceLoop, run_coupled
from mine_sandbox.engines import ENGINES
from mine_sandbox.mines import MineFileError, load_mine, mine_statistics, scenario_from_mine


def main(argv=None):
    """Run every mine file in a folder (optionally in parallel), then write a comparison table and
    figure.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder (or glob) of mine files")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--engine", choices=list(ENGINES), default="fv")
    ap.add_argument("--device", default="cpu", help="fv engine: cpu (default), auto, cuda:0, ...")
    ap.add_argument("--jobs", type=int, default=1, help="mines to run at once, each in its own process")
    ap.add_argument("--T-in", type=float, default=None, help="inlet temperature (default: from each file)")
    ap.add_argument("--flow", type=float, default=None, help="flow m3/h (default: from each file)")
    ap.add_argument("--load-mw", type=float, default=None, help="closed loop with this TESPy heat load")
    ap.add_argument("--out", default=os.path.join("figures", "mines"))
    args = ap.parse_args(argv)

    pattern = os.path.join(args.folder, "*.*") if os.path.isdir(args.folder) else args.folder
    files = sorted(f for f in glob.glob(pattern) if f.lower().endswith((".json", ".yaml", ".yml")))
    if not files:
        raise SystemExit(f"no mine files found in {args.folder}")
    engine = ENGINES[args.engine]
    if engine.check():
        raise SystemExit(f"{engine.label} is not available: {engine.check()}")
    os.makedirs(args.out, exist_ok=True)

    t_all = time.time()
    results = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run_one, path, vars(args)): path for path in files}
            for fut in as_completed(futures):
                _report(futures[fut], fut.result(), results)
    else:
        for path in files:
            _report(path, _run_one(path, vars(args)), results)
    done = [results[p] for p in files if results.get(p)]
    if not done:
        raise SystemExit("no mine could be run")
    rows = [row for row, _ in done]
    curves = {row["mine"]: curve for row, curve in done}

    keys = list(rows[0])
    with open(os.path.join(args.out, "comparison.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")
    _plot(curves, os.path.join(args.out, "comparison.png"))
    print(f"\nWrote {args.out}/comparison.csv and comparison.png  ({time.time() - t_all:.1f} s in all)")


def _run_one(path: str, a: dict):
    """One mine, start to finish. Returns (row, curve) or None; runs in a worker process with --jobs."""
    try:
        mine = load_mine(path)
    except MineFileError as e:
        return ("skip", str(e))
    sc = scenario_from_mine(mine, sim_days=a["days"], dt_days=a["dt"], flow_m3_h=a["flow"],
                            T_in_C=a["T_in"], workspace=os.path.join("mf6_run", "mines", *([str(os.getpid())] if a["jobs"] > 1 else [])))
    ctrl = (TespySurfaceLoop.for_scenario(sc, FixedHeatLoad(a["load_mw"] * 1e6)) if a["load_mw"]
            else ConstantInlet(sc.operation.T_in_C))
    engine = ENGINES[a["engine"]]
    t0 = time.time()
    r = run_coupled(sc, ctrl, model_factory=lambda p: engine.factory(p, device=a["device"]))
    st = mine_statistics(r.graph)
    i30 = r.at_day(min(30.0, r.times_d[-1]))
    row = dict(mine=mine.name, file=os.path.basename(path), cells=st["cells"],
               open_area_m2=st.get("open_area_m2", np.nan), wall_area_m2=st["wall_area_m2"],
               water_volume_m3=st["water_volume_m3"], T_rock_C=sc.ground.T_rock_C,
               heat_day30_MW=r.heat_removed_W[i30] / 1e6, heat_end_MW=r.heat_removed_W[-1] / 1e6,
               W_per_m2_wall_end=r.heat_removed_W[-1] / st["wall_area_m2"],
               effectiveness_day30_pct=100 * r.effectiveness[i30],
               effectiveness_end_pct=100 * r.effectiveness[-1],
               T_out_end_C=r.T_out_C[-1], T_in_end_C=r.T_in_C[-1], runaway=r.runaway,
               seconds=round(time.time() - t0, 1))
    # only what the plot needs: the full temperature history is far too big to send between processes
    curve = dict(times_d=r.times_d, heat_W=r.heat_removed_W, effectiveness=r.effectiveness,
                 wall_area_m2=st["wall_area_m2"])
    return row, curve


def _report(path, result, results):
    """Print one mine's headline result as it finishes and keep it for the comparison; skipped
    files are just reported.
    """
    if result and result[0] == "skip":
        print(f"SKIP {result[1]}")
        return
    row, _ = result
    results[path] = result
    print(f"{row['mine']:28s} {row['cells']:7,d} cells  day 30: {row['heat_day30_MW']:.3f} MW "
          f"({row['effectiveness_day30_pct']:.1f}%)  end: {row['heat_end_MW']:.3f} MW "
          f"({row['W_per_m2_wall_end']:.1f} W/m² wall){'  RUNAWAY' if row['runaway'] else ''}  "
          f"[{row['seconds']} s]")


def _plot(curves, path):
    """Overlay every mine's heat removed, cooling achieved and heat per square metre of wall over
    time.

    Heat per square metre of wall puts mines of different sizes on an equal footing, since the
    wall is where heat crosses into the rock.
    """
    from mine_sandbox.plotting import SERIES, plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    names = list(curves)
    for k, name in enumerate(names):
        c = curves[name]
        color = SERIES[k % len(SERIES)] if k < len(SERIES) else "#9a9a96"
        axs[0].plot(c["times_d"], c["heat_W"] / 1e6, color=color, label=name)
        axs[1].plot(c["times_d"], 100 * c["effectiveness"], color=color, label=name)
        axs[2].plot(c["times_d"], c["heat_W"] / c["wall_area_m2"], color=color, label=name)
    for ax, title, unit in ((axs[0], "Heat removed by each mine", "MW"),
                            (axs[1], "Cooling achieved", "% of possible"),
                            (axs[2], "Heat per square metre of wall", "W/m²")):
        ax.set(title=title + " (scaled after day 15)", xlabel="day", ylabel=unit)
        late = [np.max(ln.get_ydata()[np.asarray(ln.get_xdata()) > 15]) for ln in ax.get_lines()
                if np.any(np.asarray(ln.get_xdata()) > 15)]
        if late:
            ax.set_ylim(0, 1.3 * max(late))
    axs[0].legend(fontsize=8, title=None if len(names) <= 8 else "first 8 coloured")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == "__main__":
    main()

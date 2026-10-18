"""How fast is the heat solve on this machine, on the CPU and on the GPU?

    python run_benchmark.py                          # small / medium / large generated mines
    python run_benchmark.py --sizes large --steps 200
    python run_benchmark.py --mine examples/mines/traced_drifts.json

For each model it builds the fv engine on every device found, times the build
and the closed-loop steps, and checks the GPU's temperatures against the CPU's.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from mine_sandbox import ConstantInlet, CoupledStepper, FiniteVolumeModel, assemble
from mine_sandbox.compute import detect_devices
from mine_sandbox.mines import generate_mine, load_mine, scenario_from_mine


def bench(problem, device: str, steps: int):
    """Build the finite-volume model on one device and time it.

    Returns (model, build seconds, seconds per closed-loop step). The first step is run before
    timing starts, since it pays one-off costs like memory allocation and GPU kernel
    compilation.
    """
    t0 = time.perf_counter()
    m = FiniteVolumeModel(problem, device=device)
    build = time.perf_counter() - t0
    st = CoupledStepper(m, ConstantInlet(problem.scenario.operation.T_in_C))
    st.step()                                   # first step outside the timing (allocations, caches)
    t0 = time.perf_counter()
    for _ in range(steps):
        if st.step() is None:
            break
    per = (time.perf_counter() - t0) / steps
    return m, build, per


def main(argv=None):
    """Time every model on every device found and, with a GPU, report the speed-up and how closely
    its temperatures match the CPU's.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"],
                    choices=["small", "medium", "large"])
    ap.add_argument("--mine", action="append", default=[], help="mine file(s) to time as well")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)

    rep = detect_devices()
    print("Devices:", ", ".join(d.label for d in rep.devices))
    if rep.gpu_reason:
        print("  no GPU:", rep.gpu_reason)
    mines = [(f"generated {s}", generate_mine(seed=args.seed, size=s)) for s in args.sizes]
    mines += [(path, load_mine(path)) for path in args.mine]
    devices = ["cpu"] + [d.key for d in rep.gpus]
    print(f"\n{'model':28s} {'cells':>9s}  " + "  ".join(f"{d + ' build / step':>24s}" for d in devices)
          + ("  speed-up  max|ΔT|" if len(devices) > 1 else ""))
    for name, mine in mines:
        sc = scenario_from_mine(mine, sim_days=10_000)
        p = assemble(sc)
        out, fields = [], []
        for d in devices:
            m, build, per = bench(p, d, args.steps)
            out.append((build, per))
            fields.append(m.temperatures())
        line = f"{name[:28]:28s} {p.graph.n_cells:9,d}  " + "  ".join(
            f"{b:8.2f} s / {1e3 * s:7.2f} ms" for b, s in out)
        if len(devices) > 1:
            line += f"  {out[0][1] / out[1][1]:7.1f}x  {np.max(np.abs(fields[1] - fields[0])):.1e} °C"
        print(line)


if __name__ == "__main__":
    main()

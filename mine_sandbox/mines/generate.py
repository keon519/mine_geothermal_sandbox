"""Procedurally generate synthetic room-and-pillar mines (reproducible from a seed).

Each mine is a set of main entries with production panels branching off both
sides, a few roof falls blocking crosscuts, and occasionally a sealed panel
(still flooded but cut off from the flow). Dimensions are drawn from ranges
typical of Appalachian room-and-pillar coal mining; every opening is a whole
number of plan cells so the grid represents the plan exactly.

    python -m mine_sandbox.mines --count 5 --seed 1 --size medium --out mines/

Generated files are ordinary mine files: edit them, or use them as templates
for real mines.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..geology import Layer
from ..materials import ROCKS
from .io import save_mine
from .layouts import Composite, Rectangle, RoomAndPillar
from .spec import ConditionsSpec, MineGridSpec, MineSpec, SeamSpec, StrataSpec

SIZES = {   # (main pillars along, panels, panel pillars along range)
    "small": (6, (1, 2), (3, 6)),
    "medium": (10, (2, 4), (4, 9)),
    "large": (16, (4, 6), (6, 12)),
}


def generate_mine(seed: int = 0, size: str = "medium", name: str | None = None) -> MineSpec:
    """Build a random but realistic room-and-pillar mine from a seed.

    The same seed and size always give the same mine. It lays out main entries along x, adds
    production panels branching off both sides at crosscut positions (without overlapping),
    scatters roof falls through the panels, sometimes seals one panel off, and stacks up
    plausible roof and floor strata around the seam.
    """
    rng = np.random.default_rng(seed)
    main_pillars, panel_range, panel_len_range = SIZES[size]

    w = float(rng.choice([5.0, 5.5, 6.0]))                          # entry width = plan cell
    pillar = lambda: w * int(rng.integers(2, 6))                     # 2-5 entry widths (10-30 m)
    thickness = float(np.round(rng.uniform(1.0, 2.4), 1))
    depth = float(np.round(rng.uniform(120, 550), -1))

    # mains along +x
    n_main = int(rng.integers(3, 7))
    mains = RoomAndPillar(origin=(0.0, 0.0), rotation_deg=0.0, n_entries=n_main,
                          n_pillars_along=main_pillars, entry_width_m=w, crosscut_width_m=w,
                          pillar_length_m=pillar(), pillar_width_m=pillar())
    pitch = mains.crosscut_width_m + mains.pillar_length_m

    #  panels off both sides, at crosscut positions
    n_panels = int(rng.integers(panel_range[0], panel_range[1] + 1))
    parts, holes, panels = [mains], [], []
    taken = {True: [], False: []}                                   # x-intervals used per side
    for k in range(4 * n_panels):                                   # a few tries per panel
        if len(panels) == n_panels:
            break
        up = bool(rng.random() < 0.5)
        n_e = int(rng.integers(3, 7))
        pl, pw = pillar(), pillar()
        width = n_e * w + (n_e - 1) * pw
        x_left = int(rng.integers(1, main_pillars)) * pitch          # panel starts at a crosscut
        if x_left + width > mains.length_m or any(x_left < hi + pw and x_left + width + pw > lo
                                                  for lo, hi in taken[up]):
            continue
        taken[up].append((x_left, x_left + width))
        if up:   # entries along +y from the top of the mains; local v runs toward -x
            panel = RoomAndPillar(origin=(x_left + width, mains.width_m), rotation_deg=90.0,
                                  n_entries=n_e, n_pillars_along=int(rng.integers(*panel_len_range)),
                                  entry_width_m=w, crosscut_width_m=w,
                                  pillar_length_m=pl, pillar_width_m=pw)
        else:    # entries along -y from the bottom of the mains; local v runs toward +x
            panel = RoomAndPillar(origin=(x_left, 0.0), rotation_deg=-90.0,
                                  n_entries=n_e, n_pillars_along=int(rng.integers(*panel_len_range)),
                                  entry_width_m=w, crosscut_width_m=w,
                                  pillar_length_m=pl, pillar_width_m=pw)
        parts.append(panel)
        panels.append(panel)
        holes += _roof_falls(rng, panel, fraction=float(rng.uniform(0.0, 0.15)))

    #  occasionally seal a panel at its mouth (it stays flooded, but no flow reaches it)
    if len(panels) > 1 and rng.random() < 0.35:
        p = panels[int(rng.integers(len(panels)))]
        holes.append(_mouth_seal(p))

    #  strata: roof shale, then alternating beds; floor underclay, then beds
    beds, z = [], thickness / 2
    roof = float(np.round(rng.uniform(1.0, 5.0), 1))
    beds.append(Layer(z, z + roof, ROCKS["shale"])); z += roof
    while z < 40:
        t = float(np.round(rng.uniform(3, 15), 1))
        beds.append(Layer(z, z + t, ROCKS[str(rng.choice(["sandstone", "siltstone", "shale"]))]))
        z += t
    beds[-1] = Layer(beds[-1].z_bottom, 1e9, beds[-1].rock)
    z = -thickness / 2
    clay = float(np.round(rng.uniform(0.5, 2.0), 1))
    beds.append(Layer(z - clay, z, ROCKS["shale"])); z -= clay
    beds.append(Layer(-1e9, z, ROCKS[str(rng.choice(["sandstone", "siltstone"]))]))

    return MineSpec(
        name=name or f"Synthetic {size} mine {seed}",
        description=(f"Procedurally generated ({size}): {n_main} main entries, {len(panels)} panels, "
                     f"{len(holes)} closed zones; entries {w:g} m, seam {thickness:g} m at {depth:g} m."),
        source=f"mine_sandbox.mines.generate seed={seed} size={size}",
        layout=Composite(parts=tuple(parts), holes=tuple(holes)),
        seam=SeamSpec(thickness_m=thickness, depth_m=depth, rock=ROCKS["coal"]),
        strata=StrataSpec(background=ROCKS["shale"], beds=tuple(beds)),
        conditions=ConditionsSpec(),
        grid=MineGridSpec(plan_cell_m=w),
    )


def _local_to_world(panel: RoomAndPillar, u: float, v: float):
    """Convert a point in a panel's own (along, across) frame to mine plan coordinates."""
    a = np.radians(panel.rotation_deg)
    return (panel.origin[0] + u * np.cos(a) - v * np.sin(a), panel.origin[1] + u * np.sin(a) + v * np.cos(a))


def _roof_falls(rng, panel: RoomAndPillar, fraction: float) -> list:
    """Block random crosscut segments between two entries (never an entry, so the
    panel stays connected)."""
    w, c = panel.entry_width_m, panel.crosscut_width_m
    falls = []
    for i in range(1, panel.n_pillars_along + 1):          # skip the mouth crosscut
        for j in range(panel.n_entries - 1):
            if rng.random() < fraction:
                u0 = i * (c + panel.pillar_length_m)
                v0 = j * (w + panel.pillar_width_m) + w
                x, y = _local_to_world(panel, u0 + 0.25 * c, v0 + 0.25 * panel.pillar_width_m)
                falls.append(Rectangle(origin=(float(x), float(y)), length_m=0.5 * c,
                                       width_m=0.5 * panel.pillar_width_m, rotation_deg=panel.rotation_deg))
    return falls


def _mouth_seal(panel: RoomAndPillar) -> Rectangle:
    """A block across a panel's entrance, so the panel stays flooded but no flow reaches it."""
    x, y = _local_to_world(panel, 0.0, 0.0)
    return Rectangle(origin=(float(x), float(y)), length_m=panel.crosscut_width_m + 0.5 * panel.pillar_length_m,
                     width_m=panel.width_m, rotation_deg=panel.rotation_deg)


def main(argv=None):
    """Command-line entry point: generate a batch of mines and save each one as a mine file."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="first seed; files use seed, seed+1, ...")
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--out", default="mines")
    args = ap.parse_args(argv)
    for s in range(args.seed, args.seed + args.count):
        m = generate_mine(seed=s, size=args.size)
        path = os.path.join(args.out, f"synthetic_{args.size}_{s}.json")
        save_mine(m, path)
        print(f"{path}: {m.description}")


if __name__ == "__main__":
    main()

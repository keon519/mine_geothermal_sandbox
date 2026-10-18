"""MineSpec -> Scenario, so a mine file plugs into every runner, engine and the GUI."""
from __future__ import annotations

import re
from dataclasses import replace

from ..geometry import DriftSpec, RectangularSection
from ..scenario import GroundSpec, NumericsSpec, OperatingSpec, Scenario
from .grid import MineGrid
from .spec import MineSpec


def scenario_from_mine(mine: MineSpec, *, sim_days: float = 365.0, dt_days: float = 0.5,
                       flow_m3_h: float | None = None, T_in_C: float | None = None,
                       T_rock_C: float | None = None, h_film_W_m2K: float | None = None,
                       plan_cell_m: float | None = None, adv_scheme: str = "UPSTREAM",
                       name: str | None = None, workspace: str = "mf6_run") -> Scenario:
    """Turn a mine file into a Scenario that any engine, script or the GUI can run.

    Everything comes from the mine file unless overridden here, which is handy for sweeps (e.g.
    trying several flows on the same mine). The rock temperature defaults to what the seam depth
    and geothermal gradient imply.
    """
    if plan_cell_m is not None:
        mine = replace(mine, grid=replace(mine.grid, plan_cell_m=plan_cell_m))
    c = mine.conditions
    cell = mine.grid.plan_cell_m or mine.layout.min_opening_m
    return Scenario(
        # the drift spec only carries the water porosity and a typical opening for summaries
        drift=DriftSpec(length_m=1.0, section=RectangularSection(cell, mine.seam.thickness_m),
                        porosity=c.water_porosity),
        operation=OperatingSpec(flow_m3_h=flow_m3_h if flow_m3_h is not None else c.flow_m3_h,
                                T_in_C=T_in_C if T_in_C is not None else c.T_in_C),
        ground=GroundSpec(geology=mine.geology(),
                          T_rock_C=T_rock_C if T_rock_C is not None else mine.T_rock_C,
                          h_film_W_m2K=h_film_W_m2K if h_film_W_m2K is not None else c.h_film_W_m2K),
        numerics=NumericsSpec(sim_days=sim_days, dt_days=dt_days, adv_scheme=adv_scheme),
        grid=MineGrid(mine),
        name=name or re.sub(r"[^a-z0-9_-]+", "_", mine.name.lower()).strip("_")[:40] or "mine",
        workspace=workspace,
    )

"""What the user picked in the settings panel, and how it becomes a run.

LiveConfig is plain data (no Qt). build_scenario() and build_controller()
turn it into the same Scenario and InletController objects the scripts use,
so the GUI cannot drift away from the batch tools.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..grids import RadialDriftGrid, StructuredDriftGrid
from ..mines import MineSpec, scenario_from_mine
from ..scenario import Scenario, make_scenario
from ..surface import ConstantInlet, FixedHeatLoad, InletController, TespySurfaceLoop

DRIFT_GRIDS = {
    "radial": ("Radial (1-D drift + rings)", RadialDriftGrid),
    "slab2d": ("2-D slab (rock above/below)", StructuredDriftGrid.plane_2d),
    "block3d": ("3-D block (rock all round)", StructuredDriftGrid.block_3d),
}
SURFACE_MODES = {
    "fixed_inlet": "Fixed inlet temperature",
    "fixed_load": "Fixed heat load (TESPy)",
}


@dataclass(frozen=True)
class LiveConfig:
    """Everything the settings panel lets you choose, as plain frozen data.

    Keeping it free of Qt means it can be compared (to tell whether a rebuild is needed), sent
    to the simulation process, and used in tests without a window.
    """
    # workings: a single drift, or a mine
    source: str = "drift"                 # "drift" | "mine"
    drift_grid: str = "block3d"
    length_m: float = 500.0
    width_m: float = 6.0
    height_m: float = 2.0
    rock: str = "sandstone"
    mine: MineSpec | None = None
    plan_cell_m: float | None = None      # mine grid override (None: from file)
    # operation
    flow_m3_h: float = 360.0
    T_in_C: float = 35.0
    surface: str = "fixed_inlet"
    load_MW: float = 0.3
    T_max_C: float = 95.0
    # ground
    T_rock_C: float | None = 13.0         # None: from the mine file (depth + gradient)
    h_film_W_m2K: float = 800.0
    # run
    engine: str = "fv"
    device: str = "auto"                  # "auto" (GPU if one works) | "cpu" | "cuda:0", ...
    solver: str = "auto"                  # fv engine: "auto" | "block" | "direct" | "iterative"
    sim_days: float = 365.0
    dt_days: float = 0.5
    workspace: str = "mf6_run/live"


def build_scenario(c: LiveConfig) -> Scenario:
    """Turn the settings into a Scenario: from the loaded mine, or a single drift on the chosen grid."""
    if c.source == "mine":
        if c.mine is None:
            raise ValueError("open or generate a mine first")
        return scenario_from_mine(c.mine, sim_days=c.sim_days, dt_days=c.dt_days, flow_m3_h=c.flow_m3_h,
                                  T_in_C=c.T_in_C, T_rock_C=c.T_rock_C, h_film_W_m2K=c.h_film_W_m2K,
                                  plan_cell_m=c.plan_cell_m, workspace=c.workspace)
    if c.drift_grid not in DRIFT_GRIDS:
        raise ValueError(f"unknown grid '{c.drift_grid}'")
    grid = DRIFT_GRIDS[c.drift_grid][1]()
    return make_scenario(flow_m3_h=c.flow_m3_h, T_in_C=c.T_in_C, length_m=c.length_m,
                         width_m=c.width_m, height_m=c.height_m, diameter_m=None, rock=c.rock,
                         T_rock_C=13.0 if c.T_rock_C is None else c.T_rock_C,
                         h_film_W_m2K=c.h_film_W_m2K, sim_days=c.sim_days, dt_days=c.dt_days,
                         grid=grid, name="drift", workspace=c.workspace)


def build_controller(c: LiveConfig, scenario: Scenario) -> InletController:
    """Turn the settings into a surface loop: a fixed heat load through TESPy, or a fixed inlet
    temperature.
    """
    if c.surface == "fixed_load":
        return TespySurfaceLoop.for_scenario(scenario, FixedHeatLoad(c.load_MW * 1e6))
    return ConstantInlet(c.T_in_C)

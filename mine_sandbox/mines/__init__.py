"""Whole-mine models: describe a mine in a file, build a 3-D grid from it.

    spec       MineSpec and its parts (pure data)
    layouts    plan shapes: room_and_pillar, polylines, polygon, rectangle, composite
    io         load_mine / save_mine (JSON, or YAML with PyYAML), strict validation
    grid       MineGrid: MineSpec -> CellGraph (voxels; seam layer holds the water)
    scenario   scenario_from_mine: MineSpec -> Scenario for any runner or the GUI
    generate   procedural synthetic mines (python -m mine_sandbox.mines --help)
"""
from .generate import generate_mine
from .grid import MineGeometryError, MineGrid, mine_statistics
from .io import MineFileError, load_mine, mine_from_dict, mine_to_dict, read_layout, save_mine
from .layouts import (LAYOUT_TYPES, Composite, Layout, Polygon, Polylines, Rectangle, RoomAndPillar,
                      TracedDrift, register_layout)
from .scenario import scenario_from_mine
from .spec import ConditionsSpec, MineGridSpec, MineSpec, PumpSpec, SeamSpec, StrataSpec

__all__ = ["generate_mine", "MineGeometryError", "MineGrid", "mine_statistics", "MineFileError",
           "load_mine", "mine_from_dict", "mine_to_dict", "read_layout", "save_mine", "LAYOUT_TYPES",
           "Composite", "Layout", "Polygon", "Polylines", "Rectangle", "RoomAndPillar", "TracedDrift",
           "register_layout", "scenario_from_mine", "ConditionsSpec", "MineGridSpec", "MineSpec",
           "PumpSpec", "SeamSpec", "StrataSpec"]

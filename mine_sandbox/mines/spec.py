"""A mine, described as plain data - what a mine file holds.

    MineSpec
      layout      plan of the mined-out area                (layouts.py)
      seam        thickness, depth and rock of the mined seam
      strata      rock beds above and below the seam (z measured from seam centre, + up)
      conditions  loop flow, inlet temperature, rock temperature, wall film
      pumps       where Pump In and Pump Out are (or None: farthest-apart open points)
      grid        how finely to discretise it

No behaviour beyond simple derived values: turning a MineSpec into cells is
MineGrid's job, reading/writing files is io.py's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..geology import Layer, LayeredGeology
from ..materials import ROCKS, Rock
from .layouts import Layout


@dataclass(frozen=True)
class SeamSpec:
    """The mined coal seam: how thick it is, how deep it lies, and what it's made of."""
    thickness_m: float = 2.0
    depth_m: float | None = None          # below surface; used for T_rock if not given
    rock: Rock = ROCKS["coal"]

    def __post_init__(self):
        if self.thickness_m <= 0:
            raise ValueError("seam thickness_m must be positive")


@dataclass(frozen=True)
class StrataSpec:
    """Rock around the seam. Beds are Layer(z_bottom, z_top, rock) relative to the
    seam centre (z = 0), positive up; anything not covered is `background`."""
    background: Rock = ROCKS["shale"]
    beds: tuple[Layer, ...] = ()


@dataclass(frozen=True)
class ConditionsSpec:
    """How the loop is operated and how warm the ground is.

    The rock temperature can be given directly, or left as None to be worked out from the seam
    depth using the surface temperature and geothermal gradient.
    """
    flow_m3_h: float = 360.0
    T_in_C: float = 35.0
    T_rock_C: float | None = None         # None: from depth and the geothermal gradient
    T_surface_C: float = 12.0
    gradient_C_per_km: float = 25.0
    h_film_W_m2K: float = 800.0
    water_porosity: float = 1.0           # < 1 if workings are partly rubble-filled


@dataclass(frozen=True)
class PumpSpec:
    """Where Pump In and Pump Out sit, in plan coordinates. None means place them automatically at
    the two open points farthest apart.
    """
    inlet_xy: tuple | None = None
    outlet_xy: tuple | None = None


@dataclass(frozen=True)
class MineGridSpec:
    """How finely to cut the mine into cells.

    In plan, the workings get uniform cells and the solid rock around them gets cells that grow
    outward. Vertically, the seam is one layer and the roof and floor rock get layers that start
    thin at the openings and grow away from them. The extents should reach well beyond how far
    heat will spread during the run, because the outer boundary is held at the undisturbed rock
    temperature.
    """
    plan_cell_m: float | None = None      # None: the narrowest opening in the layout
    margin_m: float = 40.0                # solid rock around the workings, in plan
    margin_cells: int = 4
    layers_above: int = 6                 # rock layers above the seam ...
    layers_below: int = 5
    extent_above_m: float = 40.0          # ... reaching this far: the held-at-T_rock boundary should sit
    extent_below_m: float = 40.0          # well beyond how far heat spreads in the run (~11 m per year)
    first_layer_m: float = 0.5            # thickness of the rock layer touching roof/floor
    bridge_diagonals: bool = True         # keep diagonal tunnels continuous on the square grid


@dataclass(frozen=True)
class MineSpec:
    """A whole mine described as plain data: everything a mine file holds.

    It combines the plan of the workings, the seam, the surrounding strata, the operating
    conditions, the pump locations and the grid resolution. MineGrid turns it into cells and
    io.py reads and writes it.
    """
    name: str
    layout: Layout
    seam: SeamSpec = field(default_factory=SeamSpec)
    strata: StrataSpec = field(default_factory=StrataSpec)
    conditions: ConditionsSpec = field(default_factory=ConditionsSpec)
    pumps: PumpSpec = field(default_factory=PumpSpec)
    grid: MineGridSpec = field(default_factory=MineGridSpec)
    description: str = ""
    source: str = ""

    @property
    def T_rock_C(self) -> float:
        """Undisturbed rock temperature: the given value, or surface temperature plus the
        geothermal gradient times seam depth.
        """
        c = self.conditions
        if c.T_rock_C is not None:
            return c.T_rock_C
        if self.seam.depth_m is None:
            raise ValueError(f"mine '{self.name}': give conditions.T_rock_C or seam.depth_m")
        return c.T_surface_C + c.gradient_C_per_km * self.seam.depth_m / 1000.0

    def geology(self) -> LayeredGeology:
        """Strata plus the seam itself (the seam wins where they overlap)."""
        half = 0.5 * self.seam.thickness_m
        return LayeredGeology(background=self.strata.background,
                              layers=self.strata.beds + (Layer(-half, half, self.seam.rock),))

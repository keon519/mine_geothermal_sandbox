"""What to simulate - grouped by concern, with no solver code.

    DriftSpec      the tunnel's size and shape           (geometry.py)
    OperatingSpec  how the loop is run: flow and inlet temperature
    GroundSpec     what surrounds the drift: geology, rock temperature, wall film
    NumericsSpec   simulated time, time step, advection scheme
    Grid           how space is discretised (radial, 2-D, 3-D ...)

`Scenario` bundles them. `make_scenario()` builds one from the flat list of
knobs the project varies (flow, T_in, size, rock), which keeps sweep scripts
short.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geology import Geology, UniformGeology, as_geology
from .geometry import CircularSection, CrossSection, DriftSpec, RectangularSection
from .grids import Grid, RadialDriftGrid
from .materials import ROCKS, WATER

SEC_PER_DAY = 86400.0


@dataclass(frozen=True)
class OperatingSpec:
    """How the surface loop runs the mine: how much water is pumped through and how hot it goes in."""
    flow_m3_h: float = 360.0      # water pumped through the drift (360 m3/h = 100 L/s)
    T_in_C: float = 35.0          # inlet temperature (used when the inlet is held fixed)

    @property
    def flow_m3_d(self) -> float:
        """Pumping rate converted to cubic metres per day, the unit MODFLOW works in."""
        return self.flow_m3_h * 24.0

    @property
    def mass_flow_kg_s(self) -> float:
        """Pumping rate as a mass flow, which is what heat-balance equations (m_dot * cp * dT) need."""
        return self.flow_m3_h / 3600.0 * WATER.rho


@dataclass(frozen=True)
class GroundSpec:
    """Everything about the ground around the workings.

    The rock starts at a uniform undisturbed temperature, and the wall film describes how easily
    heat passes from moving water into the rock face. A tiny rock porosity is kept so that every
    cell has some water in it, which keeps the transport equations well behaved even in "solid"
    rock.
    """
    geology: Geology = UniformGeology(ROCKS["sandstone"])
    T_rock_C: float = 13.0        # undisturbed rock temperature
    h_film_W_m2K: float = 800.0   # water -> wall convective coefficient
    rock_porosity: float = 0.01   # pore water in the rock (keeps storage well-posed)


@dataclass(frozen=True)
class NumericsSpec:
    """How the simulation is stepped in time.

    Shorter time steps track fast changes more accurately but take longer to run. The advection
    scheme decides how heat is carried along with the water: upstream weighting is robust but
    smears sharp temperature fronts, while TVD keeps fronts sharper at a little extra cost.
    """
    sim_days: float = 365.0
    dt_days: float = 0.5
    adv_scheme: str = "UPSTREAM"  # "UPSTREAM" (robust) or "TVD" (sharper fronts)


@dataclass(frozen=True)
class InletPeriod:
    """A stretch of time with constant inlet conditions."""
    length_d: float
    T_in_C: float
    flow_m3_d: float


@dataclass(frozen=True)
class Scenario:
    """A complete description of one simulation, with no solver code in it.

    It groups the drift shape, how the loop is operated, the ground, the time stepping, and the
    grid that cuts space into cells. Because it's frozen and solver-neutral, the same Scenario
    can be handed to MODFLOW or the built-in engine and they will solve the same problem.
    """
    drift: DriftSpec = field(default_factory=DriftSpec)
    operation: OperatingSpec = field(default_factory=OperatingSpec)
    ground: GroundSpec = field(default_factory=GroundSpec)
    numerics: NumericsSpec = field(default_factory=NumericsSpec)
    grid: Grid = field(default_factory=RadialDriftGrid)
    name: str = "mine"
    workspace: str = "mf6_run"

    def default_schedule(self) -> list[InletPeriod]:
        """A single period that holds the configured inlet temperature and flow for the whole run."""
        return [InletPeriod(self.numerics.sim_days, self.operation.T_in_C, self.operation.flow_m3_d)]

    def summary(self) -> str:
        """One-line description of the scenario for logs and figure titles."""
        o, g = self.operation, self.ground
        return (f"{self.name}: {self.grid.describe_workings(self.drift)}, Q={o.flow_m3_h:.0f} m3/h "
                f"({o.mass_flow_kg_s:.1f} kg/s), T_in={o.T_in_C:.1f} C, rock={g.geology.label}, "
                f"T_rock={g.T_rock_C:.1f} C, grid={self.grid.label}")


def make_scenario(*, flow_m3_h: float = 360.0, T_in_C: float = 35.0,
                  length_m: float = 500.0, diameter_m: float | None = 4.0,
                  width_m: float | None = None, height_m: float | None = None,
                  rock="sandstone", T_rock_C: float = 13.0, h_film_W_m2K: float = 800.0,
                  drift_porosity: float = 1.0, sim_days: float = 365.0, dt_days: float = 0.5,
                  adv_scheme: str = "UPSTREAM", grid: Grid | None = None,
                  name: str = "mine", workspace: str = "mf6_run") -> Scenario:
    """Flat constructor for the common case.

    Give `diameter_m` for a round drift, or `width_m` + `height_m` for a
    rectangular one. `rock` may be a rock name, a Rock, a (k, rho, cp) tuple,
    or a Geology (e.g. LayeredGeology)."""
    if width_m is not None and height_m is not None:
        section: CrossSection = RectangularSection(width_m, height_m)
    elif diameter_m is not None:
        section = CircularSection(diameter_m)
    else:
        raise ValueError("give diameter_m, or both width_m and height_m")
    return Scenario(
        drift=DriftSpec(length_m=length_m, section=section, porosity=drift_porosity),
        operation=OperatingSpec(flow_m3_h=flow_m3_h, T_in_C=T_in_C),
        ground=GroundSpec(geology=as_geology(rock), T_rock_C=T_rock_C, h_film_W_m2K=h_film_W_m2K),
        numerics=NumericsSpec(sim_days=sim_days, dt_days=dt_days, adv_scheme=adv_scheme),
        grid=grid or RadialDriftGrid(),
        name=name, workspace=workspace,
    )

"""Scenario -> ThermalProblem: the grid plus the physical property of every cell.

This is the one place where physics meets geometry:
  * rock properties come from the Geology at each cell centre;
  * the drift's cells get the drift porosity, rock cells the rock porosity;
  * the water -> wall film resistance is folded into the drift side of every
    drift-rock face (distance = k_water / h, so distance / k_water = 1 / h);
  * hydraulic conductivity is a numerical device: large in the water (head
    drop along the main route ~0.1 m), 1e-9 of that in rock (no flow through rock).

The result is solver-neutral: nothing here knows about MODFLOW.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import CellGraph
from .materials import WATER
from .scenario import Scenario

K_CONTRAST = 1.0e-9       # rock K / drift K


@dataclass(frozen=True)
class CellProperties:
    """The physical properties of every cell, as flat arrays indexed by cell number.

    Each cell is a mix of water (its porosity fraction) and solid rock. Heat storage blends the
    two, while conductivity uses the solid's value. Hydraulic conductivity is the knob that
    makes water flow only through the open workings and not through the rock.
    """
    porosity: np.ndarray          # (n,) water fraction of each cell
    k_solid: np.ndarray           # (n,) W/(m K)
    rho_solid: np.ndarray         # (n,) kg/m^3
    cp_solid: np.ndarray          # (n,) J/(kg K)
    hydraulic_k_m_d: np.ndarray   # (n,) m/d

    @property
    def volumetric_heat_capacity(self) -> np.ndarray:
        """theta rho_w c_w + (1 - theta) rho_s c_s   [J / (m^3 K)] per cell."""
        return (self.porosity * WATER.volumetric_heat_capacity
                + (1.0 - self.porosity) * self.rho_solid * self.cp_solid)


@dataclass(frozen=True)
class ThermalProblem:
    """A scenario turned into something a solver can work on directly.

    It pairs the grid's cell graph (with the water-to-wall film already folded into the face
    distances) with per-cell properties. Every engine takes a ThermalProblem, which is what
    guarantees they are all solving the same thing.
    """
    scenario: Scenario
    graph: CellGraph              # with the wall film already applied
    props: CellProperties

    @property
    def water_volume_m3(self) -> float:
        """Total volume of water held in the workings."""
        d = self.graph.drift_cells
        return float(np.sum(self.graph.volumes[d] * self.props.porosity[d]))

    @property
    def residence_time_d(self) -> float:
        """Water volume / flow: how long the workings take to flush once."""
        return self.water_volume_m3 / self.scenario.operation.flow_m3_d


class GeologyNotSupportedError(ValueError):
    """Raised when a grid can't represent the requested geology, e.g. layered rock on an
    axisymmetric grid.
    """
    pass


def assemble(scenario: Scenario, graph: CellGraph | None = None) -> ThermalProblem:
    """Turn a Scenario into a ThermalProblem ready for any engine.

    Builds the grid (unless one is passed in), checks it, assigns each cell its rock and water
    properties, and applies the wall film to the faces between water and rock.

    Pass `graph` if it was already built from this scenario's grid (the live session builds it
    as its own stage so it can report progress).
    """
    if graph is None:
        graph = scenario.grid.build(scenario.drift)
    graph.validate()
    geology = scenario.ground.geology
    if not graph.resolves_depth and not geology.is_uniform:
        raise GeologyNotSupportedError(
            f"{scenario.grid.label} is axisymmetric and cannot place layered rock; "
            "use StructuredDriftGrid.plane_2d() or .block_3d() for layered geology.")
    return ThermalProblem(scenario=scenario, graph=_apply_wall_film(graph, scenario),
                          props=_cell_properties(graph, scenario))


def _cell_properties(graph: CellGraph, sc: Scenario) -> CellProperties:
    """Look up the rock at every cell centre and fill in porosity, thermal and hydraulic properties
    for each cell.
    """
    rocks = sc.ground.geology.rocks_at(graph.centroids)
    drift = graph.is_drift
    k_drift = _water_conductivity(graph, sc.operation.flow_m3_d)
    return CellProperties(
        porosity=np.where(drift, sc.drift.porosity, sc.ground.rock_porosity),
        k_solid=np.array([r.k for r in rocks]),
        rho_solid=np.array([r.rho for r in rocks]),
        cp_solid=np.array([r.cp for r in rocks]),
        hydraulic_k_m_d=np.where(drift, k_drift, k_drift * K_CONTRAST),
    )


def _water_conductivity(graph: CellGraph, flow_m3_d: float) -> float:
    """K for water cells so that the head drop along the main route is <= ~0.1 m:
    dh ~ Q L / (K A) with L the route length and A the narrowest water-water face."""
    water = graph.is_drift
    ww = water[graph.conn_a] & water[graph.conn_b]
    area = graph.face_area[ww].min() if np.any(ww) else 1.0
    return max(1.0e4, 10.0 * flow_m3_d * max(graph.path_length_m, 1.0) / area)


def _apply_wall_film(graph: CellGraph, sc: Scenario) -> CellGraph:
    """Fold the water-to-wall heat transfer coefficient into the grid's face distances.

    Conduction between two cells depends on distance divided by conductivity. By setting the
    water-side distance on wall faces to k_water / h, that ratio becomes 1 / h, so the solver
    models the film resistance without needing a special boundary condition.
    """
    film = WATER.k / sc.ground.h_film_W_m2K
    drift = graph.is_drift
    a_drift, b_drift = drift[graph.conn_a], drift[graph.conn_b]
    wall = a_drift ^ b_drift                     # faces between water and rock
    dist_a = np.where(wall & a_drift, film, graph.dist_a)
    dist_b = np.where(wall & b_drift, film, graph.dist_b)
    return graph.with_distances(dist_a, dist_b)

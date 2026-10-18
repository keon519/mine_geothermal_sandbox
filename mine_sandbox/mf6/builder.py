"""ThermalProblem -> a flopy MODFLOW 6 simulation (GWF + GWE on a DISU grid).

The builder's only job is translation: it writes what the problem says and
makes no physical choices of its own. Works for any CellGraph, so 1-D, 2-D
and 3-D grids all go through the same code.

DISU trick used for every grid: each cell gets top = 1, bottom = 0 and
"area" = its true volume, so MODFLOW's volume = area x thickness is exact,
and every face is flagged horizontal (IHC = 1) so HWVA x thickness = HWVA =
the true face area.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import flopy
import numpy as np

from ..assembly import ThermalProblem
from ..grids import CellGraph
from ..materials import WATER
from ..scenario import SEC_PER_DAY, InletPeriod
from .executables import find_mf6

GWF_NAME, GWE_NAME, WEL_NAME = "shaftgwf", "shaftgwe", "WEL-1"   # MF6 caps names at 16 chars


@dataclass(frozen=True)
class SolverSettings:
    """Convergence settings for MODFLOW's linear solvers, one set for the flow model and one for
    the heat model.
    """
    flow: dict = field(default_factory=lambda: dict(
        complexity="SIMPLE", outer_dvclose=1e-5, inner_dvclose=1e-6, outer_maximum=100,
        linear_acceleration="BICGSTAB"))
    heat: dict = field(default_factory=lambda: dict(
        complexity="MODERATE", outer_dvclose=1e-5, inner_dvclose=1e-6, outer_maximum=300,
        inner_maximum=200, linear_acceleration="BICGSTAB"))


@dataclass
class Mf6Handles:
    """The pieces of a built MODFLOW simulation that callers need to run it and read its output."""
    sim: flopy.mf6.MFSimulation
    gwf: flopy.mf6.ModflowGwf
    gwe: flopy.mf6.ModflowGwe
    workspace: str
    schedule: list


def disu_arrays(graph: CellGraph) -> dict:
    """Translate a CellGraph into MODFLOW's unstructured-grid (DISU) arrays.

    MODFLOW wants, for each cell, a list of its neighbours with the distance to each shared face
    and that face's area. The graph stores each face once, so this expands every face into both
    directions and packs the result in MODFLOW's compressed-row layout, with each cell listed
    first in its own row as MODFLOW requires.
    """
    nbrs = [[] for _ in range(graph.n_cells)]
    for a, b, da, db, ar, ang in zip(graph.conn_a, graph.conn_b, graph.dist_a, graph.dist_b,
                                     graph.face_area, graph.angle_deg):
        nbrs[a].append((b, da, ar, ang))
        nbrs[b].append((a, db, ar, (ang + 180.0) % 360.0))
    iac, ja, cl12, hwva, ang = [], [], [], [], []
    for n, row in enumerate(nbrs):
        row.sort()
        iac.append(1 + len(row))
        ja.append(n); cl12.append(0.0); hwva.append(0.0); ang.append(0.0)
        for b, d, ar, a in row:
            ja.append(b); cl12.append(d); hwva.append(ar); ang.append(a)
    lay = graph.layout
    vertices = [(i, x, y) for i, (x, y) in enumerate(lay.vertices.tolist())]
    cell2d = [(i, cx, cy, 4, *map(int, vs))
              for i, ((cx, cy), vs) in enumerate(zip(lay.centres.tolist(), lay.cells.tolist()))]
    return dict(nodes=graph.n_cells, nja=len(ja), top=1.0, bot=0.0, area=graph.volumes,
                iac=iac, ja=ja, ihc=[1] * len(ja), cl12=cl12, hwva=hwva, angldegx=ang,
                nvert=len(vertices), vertices=vertices, cell2d=cell2d)


class Mf6SimulationBuilder:
    """Translates a ThermalProblem into a flopy MODFLOW 6 simulation: a flow model coupled to a
    heat model.
    """
    def __init__(self, exe: str | None = None, solver: SolverSettings | None = None):
        self.exe = exe or find_mf6()
        self.solver = solver or SolverSettings()

    def build(self, problem: ThermalProblem, schedule: list[InletPeriod], workspace: str,
              T_init: np.ndarray | None = None) -> Mf6Handles:
        """Create the simulation in `workspace`: time discretisation, both models, and the exchange
        that feeds flows to the heat model.
        """
        os.makedirs(workspace, exist_ok=True)
        sim = flopy.mf6.MFSimulation(sim_name="mine", sim_ws=workspace, exe_name=self.exe)
        dt = problem.scenario.numerics.dt_days
        flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(schedule),
                              perioddata=[(p.length_d, max(1, int(np.ceil(p.length_d / dt))), 1.0)
                                          for p in schedule])
        disu = disu_arrays(problem.graph)
        gwf = self._flow_model(sim, problem, schedule, disu)
        gwe = self._heat_model(sim, problem, disu, T_init)
        flopy.mf6.ModflowGwfgwe(sim, exgtype="GWF6-GWE6", exgmnamea=GWF_NAME, exgmnameb=GWE_NAME)
        return Mf6Handles(sim=sim, gwf=gwf, gwe=gwe, workspace=workspace, schedule=schedule)

    #  GWF
    def _flow_model(self, sim, problem, schedule, disu):
        """Groundwater flow model: water injected at Pump In and drawn off at a fixed head at Pump
        Out.

        The injection well carries the inlet temperature as an auxiliary value, which is how the
        heat model learns what temperature the incoming water has.
        """
        g = problem.graph
        gwf = flopy.mf6.ModflowGwf(sim, modelname=GWF_NAME, save_flows=True)
        flopy.mf6.ModflowGwfdisu(gwf, length_units="METERS", **disu)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=problem.props.hydraulic_k_m_d)
        flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        # Pump In: a well carrying the inlet temperature as an auxiliary variable
        flopy.mf6.ModflowGwfwel(gwf, auxiliary=["TEMPERATURE"], pname=WEL_NAME, save_flows=True,
                                stress_period_data={i: [((g.inlet_cell,), p.flow_m3_d, p.T_in_C)]
                                                    for i, p in enumerate(schedule)})
        # Pump Out: fixed head at the last drift cell
        flopy.mf6.ModflowGwfchd(gwf, pname="CHD-1", save_flows=True,
                                stress_period_data={0: [((g.outlet_cell,), 0.0)]})
        flopy.mf6.ModflowGwfoc(gwf, budget_filerecord=f"{GWF_NAME}.cbc", head_filerecord=f"{GWF_NAME}.hds",
                               saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")])
        ims = flopy.mf6.ModflowIms(sim, pname="ims_gwf", filename=f"{GWF_NAME}.ims", **self.solver.flow)
        sim.register_ims_package(ims, [GWF_NAME])
        return gwf

    #  GWE
    def _heat_model(self, sim, problem, disu, T_init):
        """Heat model: storage, conduction and advection, the inlet water's temperature as a
        source, and far-field cells held at rock temperature.
        """
        g, p, sc = problem.graph, problem.props, problem.scenario
        gwe = flopy.mf6.ModflowGwe(sim, modelname=GWE_NAME, save_flows=True)
        flopy.mf6.ModflowGwedisu(gwe, length_units="METERS", **disu)
        strt = np.full(g.n_cells, sc.ground.T_rock_C) if T_init is None else np.asarray(T_init, float)
        flopy.mf6.ModflowGweic(gwe, strt=strt)
        flopy.mf6.ModflowGweadv(gwe, scheme=sc.numerics.adv_scheme)
        # conductivities in J/day/m/K because the model runs in days
        flopy.mf6.ModflowGwecnd(gwe, xt3d_off=True, alh=0.0, ath1=0.0,
                                ktw=WATER.k * SEC_PER_DAY, kts=p.k_solid * SEC_PER_DAY)
        flopy.mf6.ModflowGweest(gwe, porosity=p.porosity, heat_capacity_water=WATER.cp,
                                density_water=WATER.rho, heat_capacity_solid=p.cp_solid,
                                density_solid=p.rho_solid, save_flows=True)
        flopy.mf6.ModflowGwessm(gwe, sources=[(WEL_NAME, "AUX", "TEMPERATURE")])
        flopy.mf6.ModflowGwectp(gwe, pname="CTP-1", stress_period_data={
            0: [((int(n),), sc.ground.T_rock_C) for n in g.far_field_cells]})
        flopy.mf6.ModflowGweoc(gwe, budget_filerecord=f"{GWE_NAME}.cbc",
                               temperature_filerecord=f"{GWE_NAME}.ucn",
                               saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "LAST")])
        ims = flopy.mf6.ModflowIms(sim, pname="ims_gwe", filename=f"{GWE_NAME}.ims", **self.solver.heat)
        sim.register_ims_package(ims, [GWE_NAME])
        return gwe

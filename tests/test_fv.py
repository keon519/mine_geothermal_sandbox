"""The built-in finite-volume engine against MODFLOW 6, and on its own."""
import numpy as np
import pytest

from conftest import TINY_MINE, needs_libmf6
from mine_sandbox import (ConstantInlet, CoupledStepper, FiniteVolumeModel, StructuredDriftGrid, assemble,
                          make_scenario, run_coupled)
from mine_sandbox.mines import scenario_from_mine


def _mf6():
    from mine_sandbox.mf6 import Mf6ApiSession
    return Mf6ApiSession


@needs_libmf6
@pytest.mark.parametrize("label,kw", [
    ("radial", {}),
    ("block3d", dict(width_m=6, height_m=2, diameter_m=None, grid=StructuredDriftGrid.block_3d(n_x=6, n_side=6))),
    ("mine", "mine"),
])
def test_matches_modflow(ws, label, kw):
    """Same discretisation; the only difference is MODFLOW's ~1e-9 Q leakage through rock."""
    sc = (scenario_from_mine(TINY_MINE, sim_days=20, workspace=ws) if kw == "mine"
          else make_scenario(workspace=ws, sim_days=20, **kw))
    a = run_coupled(sc, model_factory=_mf6())
    b = run_coupled(sc, model_factory=FiniteVolumeModel)
    assert np.max(np.abs(a.T_out_C - b.T_out_C)) < 1e-4
    assert np.max(np.abs(a.temperatures[-1] - b.temperatures[-1])) < 5e-3


def test_conserves_energy_on_a_mine():
    sc = scenario_from_mine(TINY_MINE, sim_days=10, T_rock_C=13.0)
    r = run_coupled(sc, model_factory=FiniteVolumeModel)
    dt = np.diff(np.r_[0.0, r.times_d]) * 86400.0
    assert np.sum(r.heat_removed_W * dt) == pytest.approx(r.stored_heat_J(), rel=2e-4)


def test_direct_and_iterative_solvers_agree():
    p = assemble(scenario_from_mine(TINY_MINE, sim_days=5))
    out = []
    for kind in ("direct", "iterative"):
        st = CoupledStepper(FiniteVolumeModel(p, solver=kind), ConstantInlet(35.0))
        while st.step() is not None:
            pass
        out.append(st.model.temperatures())
    assert np.max(np.abs(out[0] - out[1])) < 1e-5


def test_flow_follows_the_network():
    """All pumped water reaches Pump Out; nothing flows through rock."""
    p = assemble(scenario_from_mine(TINY_MINE, sim_days=1))
    m = FiniteVolumeModel(p)
    g, q = p.graph, m.face_flow
    out = g.outlet_cell
    arriving = q[g.conn_b == out].sum() - q[g.conn_a == out].sum()
    assert arriving == pytest.approx(p.scenario.operation.flow_m3_d, rel=1e-9)
    water = g.is_drift
    assert np.all(q[~(water[g.conn_a] & water[g.conn_b])] == 0)

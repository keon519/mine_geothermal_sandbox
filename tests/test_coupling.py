"""The coupler depends on interfaces, so implementations must be interchangeable."""
import numpy as np
import pytest

from conftest import needs_libmf6, needs_mf6
from mine_sandbox import (ConstantInlet, FixedHeatLoad, FixedInletTemperature, LoopState,
                          ThermalModel, TespySurfaceLoop, make_scenario, run_coupled,
                          run_fixed_inlet)
from mine_sandbox.mf6 import Mf6ApiSession, Mf6RestartSession


@needs_libmf6
def test_api_session_matches_batch_run(ws):
    """Same scenario, same constant inlet: stepping through the API == one batch run."""
    sc = make_scenario(workspace=ws, sim_days=20)
    batch = run_fixed_inlet(sc)
    coupled = run_coupled(sc, ConstantInlet(sc.operation.T_in_C), model_factory=Mf6ApiSession)
    assert np.allclose(coupled.T_out_C, batch.T_out_C, atol=1e-8)


@needs_mf6
def test_restart_session_is_a_drop_in_replacement(ws):
    sc = make_scenario(workspace=ws, sim_days=20)
    batch = run_fixed_inlet(sc)
    r = run_coupled(sc, ConstantInlet(sc.operation.T_in_C),
                    model_factory=lambda p: Mf6RestartSession(p, chunk_days=5.0))
    assert r.times_d[-1] == pytest.approx(20.0)
    assert r.T_out_C[-1] == pytest.approx(batch.T_out_C[-1], abs=1e-6)


@needs_libmf6
def test_runaway_guard(ws):
    sc = make_scenario(workspace=ws, sim_days=120)
    r = run_coupled(sc, TespySurfaceLoop.for_scenario(sc, FixedHeatLoad(2e6)), T_max_C=95.0)
    assert r.runaway and r.times_d[-1] < 120


def test_tespy_modes_close_the_energy_balance():
    loop = TespySurfaceLoop(FixedHeatLoad(1e6), mass_flow_kg_s=100.0)
    s = loop.solve(20.0)
    assert s.Q_load_W == pytest.approx(1e6)
    assert s.T_in_C > 20.0 + 1e6 / (100 * 4200)                    # load plus a bit of pump heat
    fixed = TespySurfaceLoop(FixedInletTemperature(35.0), mass_flow_kg_s=100.0).solve(30.0)
    assert fixed.T_in_C == pytest.approx(35.0)
    assert fixed.Q_load_W == pytest.approx(100 * 4180 * 5 - fixed.pump_power_W, rel=0.02)


class LumpedTank(ThermalModel):
    """A toy model with no MODFLOW at all: one well-mixed tank losing heat to rock.

    Each step the outlet settles at the balance point between heat brought in by the water and
    heat lost through a single conductance (UA) to the rock. Its answer is known in closed form,
    which makes it a handy stand-in for checking the coupler.
    """

    def __init__(self, problem, UA_W_K=5e4, days=10.0, dt=0.5):
        self.problem, self.UA, self.dt, self.end = problem, UA_W_K, dt, days
        self.t, self.T_in = 0.0, 0.0
        self.T = np.full(problem.graph.n_cells, problem.scenario.ground.T_rock_C)

    time_d = property(lambda s: s.t)
    end_time_d = property(lambda s: s.end)

    def set_inlet_temperature(self, T_C):
        self.T_in = T_C

    def advance(self):
        sc = self.problem.scenario
        m_cp = sc.operation.mass_flow_kg_s * 4184.0
        T_r = sc.ground.T_rock_C
        T_out = (m_cp * self.T_in + self.UA * T_r) / (m_cp + self.UA)    # steady tank
        self.T[:] = T_r
        self.T[self.problem.graph.outlet_cell] = T_out
        self.t += self.dt

    def temperatures(self):
        return self.T.copy()


def test_coupler_accepts_any_thermal_model():
    """run_coupled works with a model that has nothing to do with MODFLOW."""
    sc = make_scenario()
    r = run_coupled(sc, ConstantInlet(35.0), model_factory=LumpedTank)
    assert len(r.times_d) == 20
    assert r.T_out_C[-1] == pytest.approx((418400 * 35 + 5e4 * 13) / (418400 + 5e4))


class Ramp:
    """Any object with inlet_state() works as a controller (here: a warming schedule)."""
    def inlet_state(self, T_out_C, t_d):
        return LoopState(T_in_C=20.0 + t_d)


def test_coupler_accepts_any_controller():
    r = run_coupled(make_scenario(), Ramp(), model_factory=LumpedTank)
    assert np.allclose(np.diff(r.T_in_C), 0.5)

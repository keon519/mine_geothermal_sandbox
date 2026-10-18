"""Run MODFLOW 6 start-to-finish with a pre-set inlet schedule (no feedback).

Needs only the mf6 executable. Used for fixed-inlet studies and as the
engine inside Mf6RestartSession.
"""
from __future__ import annotations

import os

import numpy as np

from ..assembly import ThermalProblem
from ..energy import advected_heat_W
from ..materials import WATER
from ..results import RunResult
from ..scenario import SEC_PER_DAY, InletPeriod
from .builder import Mf6SimulationBuilder


class Mf6RunError(RuntimeError):
    """MODFLOW 6 stopped with an error; the message includes the tail of its output and listing file."""
    pass


class Mf6BatchRunner:
    """Writes a complete MODFLOW 6 simulation, runs the executable once, and reads the results back."""
    def __init__(self, builder: Mf6SimulationBuilder | None = None):
        self.builder = builder or Mf6SimulationBuilder()

    def run(self, problem: ThermalProblem, schedule: list[InletPeriod] | None = None,
            T_init: np.ndarray | None = None, workspace: str | None = None) -> RunResult:
        """Run the problem through MODFLOW 6 over the given inlet schedule.

        `T_init` lets a run start from an earlier temperature field instead of undisturbed rock,
        which is how the restart session strings short runs together.
        """
        sc = problem.scenario
        schedule = schedule or sc.default_schedule()
        ws = workspace or os.path.join(sc.workspace, sc.name)
        h = self.builder.build(problem, schedule, ws, T_init)
        h.sim.write_simulation(silent=True)
        ok, buff = h.sim.run_simulation(silent=True)
        if not ok:
            raise Mf6RunError(f"MODFLOW 6 failed in {ws}:\n{_error_report(ws, buff)}")
        return self._read(problem, h)

    @staticmethod
    def _read(problem: ThermalProblem, h) -> RunResult:
        """Load MODFLOW's saved temperatures and line them up with the inlet schedule to build a
        RunResult.
        """
        tobj = h.gwe.output.temperature()
        times = np.asarray(tobj.get_times())
        temps = tobj.get_alldata().reshape(len(times), -1)
        T_in, Q = np.empty_like(times), np.empty_like(times)
        t0 = 0.0
        for p in h.schedule:
            m = (times > t0 + 1e-9) & (times <= t0 + p.length_d + 1e-9)
            T_in[m], Q[m] = p.T_in_C, p.flow_m3_d
            t0 += p.length_d
        T_out = temps[:, problem.graph.outlet_cell]
        m_dot = Q / SEC_PER_DAY * WATER.rho
        return RunResult(problem=problem, times_d=times, T_in_C=T_in, T_out_C=T_out,
                         heat_removed_W=advected_heat_W(m_dot, T_in, T_out), temperatures=temps)


def _error_report(ws: str, buff) -> str:
    """Collect the last lines of MODFLOW's console output plus the error section of its listing
    file, if any.
    """
    msg = "\n".join(buff[-30:])
    lst = os.path.join(ws, "mfsim.lst")
    if os.path.exists(lst):
        txt = open(lst).read()
        if "ERROR REPORT" in txt:
            msg += "\n" + txt[txt.index("ERROR REPORT"):][:1500]
    return msg

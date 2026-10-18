"""Two interchangeable ThermalModel implementations backed by MODFLOW 6.

Mf6ApiSession      drives libmf6 one heat time step at a time through
                   modflowapi. Exact coupling; needs the shared library.
Mf6RestartSession  re-runs the mf6 executable for each chunk of time,
                   restarting from the previous temperature field. Needs only
                   the executable; the inlet is frozen for each chunk, so keep
                   chunk_days short compared with how fast things change.
"""
from __future__ import annotations

import os

import numpy as np

from ..assembly import ThermalProblem
from ..model import ThermalModel
from ..scenario import InletPeriod
from .batch import Mf6BatchRunner
from .builder import GWE_NAME, GWF_NAME, WEL_NAME, Mf6SimulationBuilder
from .executables import find_libmf6


class Mf6ApiSession(ThermalModel):
    """MODFLOW 6 driven one time step at a time through its shared library.

    The simulation is written once and loaded into libmf6. Before each step the inlet
    temperature is poked directly into MODFLOW's memory, so the surface loop can change it every
    step with no restarts and no approximation.
    """
    def __init__(self, problem: ThermalProblem, libmf6: str | None = None,
                 builder: Mf6SimulationBuilder | None = None, workspace: str | None = None):
        from modflowapi import ModflowApi            # optional dependency, imported on use
        self.problem = problem
        sc = problem.scenario
        ws = workspace or os.path.join(sc.workspace, sc.name + "_coupled")
        h = (builder or Mf6SimulationBuilder()).build(problem, sc.default_schedule(), ws)
        h.sim.write_simulation(silent=True)
        self._api = ModflowApi(libmf6 or find_libmf6(), working_directory=ws)
        self._api.initialize()
        # The inlet temperature lives in two places once the models are running:
        # the WEL package's input copy, and GWE's copy of the flow-model boundary data.
        self._aux_input = self._api.get_var_address("AUXVAR_IDM", GWF_NAME.upper(), WEL_NAME)
        self._aux_gwe = self._api.get_var_address("AUXVAR", GWE_NAME.upper(), "FMI-FT1")
        self._temp = self._api.get_var_address("X", GWE_NAME.upper())
        self._T_in = sc.operation.T_in_C
        self._open = True

    @property
    def time_d(self) -> float:
        return float(self._api.get_current_time())

    @property
    def end_time_d(self) -> float:
        return float(self._api.get_end_time())

    def set_inlet_temperature(self, T_C: float) -> None:
        self._T_in = float(T_C)

    def advance(self) -> None:
        """Write the inlet temperature into both places MODFLOW reads it from, then run one time step."""
        api = self._api
        api.get_value_ptr(self._aux_input)[0] = self._T_in
        api.prepare_time_step(api.get_time_step())
        api.get_value_ptr(self._aux_gwe)[0] = self._T_in
        api.do_time_step()
        api.finalize_time_step()

    def temperatures(self) -> np.ndarray:
        """Copy of MODFLOW's current temperature array."""
        return self._api.get_value_ptr(self._temp).copy()

    def close(self) -> None:
        """Shut down the libmf6 session (safe to call more than once)."""
        if self._open:
            self._api.finalize()
            self._open = False


class Mf6RestartSession(ThermalModel):
    """MODFLOW 6 run in short chunks with the executable, restarting from the last temperature
    field each time.

    This needs only the mf6 executable, not the shared library. The inlet temperature is held
    fixed within each chunk, so the coupling is only approximate; keep `chunk_days` short
    compared with how quickly the loop temperatures change.
    """
    def __init__(self, problem: ThermalProblem, chunk_days: float | None = None,
                 runner: Mf6BatchRunner | None = None, workspace: str | None = None):
        self.problem = problem
        sc = problem.scenario
        self.chunk_days = chunk_days or 5.0
        self._runner = runner or Mf6BatchRunner()
        self._ws = workspace or os.path.join(sc.workspace, sc.name + "_restart")
        self._t = 0.0
        self._T_in = sc.operation.T_in_C
        self._field = np.full(problem.graph.n_cells, sc.ground.T_rock_C)

    @property
    def time_d(self) -> float:
        return self._t

    @property
    def end_time_d(self) -> float:
        return self.problem.scenario.numerics.sim_days

    def set_inlet_temperature(self, T_C: float) -> None:
        self._T_in = float(T_C)

    def advance(self) -> None:
        """Run MODFLOW for one chunk at the current inlet temperature, starting from the saved field."""
        length = min(self.chunk_days, self.end_time_d - self._t)
        period = InletPeriod(length, self._T_in, self.problem.scenario.operation.flow_m3_d)
        res = self._runner.run(self.problem, [period], T_init=self._field, workspace=self._ws)
        self._field = res.temperatures[-1].copy()
        self._t += length

    def temperatures(self) -> np.ndarray:
        """Copy of the temperature field at the end of the last chunk."""
        return self._field.copy()

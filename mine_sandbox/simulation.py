"""High-level entry points. They depend on abstractions only:

    run_fixed_inlet(scenario, runner)            any object with .run(problem) -> RunResult
    run_coupled(scenario, controller, model)     any InletController + any ThermalModel
    CoupledStepper                               the loop body, shared with the live GUI

Defaults use MODFLOW 6, but nothing here imports flopy or modflowapi, so a
different solver or grid slots in by passing a different runner / factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .assembly import ThermalProblem, assemble
from .energy import advected_heat_W
from .model import ThermalModel
from .results import RunResult
from .scenario import Scenario
from .surface import ConstantInlet, InletController

ModelFactory = Callable[[ThermalProblem], ThermalModel]


class BatchRunner(Protocol):
    """Anything that can run a whole fixed-inlet simulation in one go and hand back the result."""
    def run(self, problem: ThermalProblem) -> RunResult:
        """Run the problem from start to finish and return its RunResult."""


def run_fixed_inlet(scenario: Scenario, runner: BatchRunner | None = None) -> RunResult:
    """Constant flow and inlet temperature for the whole run (no surface feedback)."""
    if runner is None:
        from .mf6 import Mf6BatchRunner
        runner = Mf6BatchRunner()
    return runner.run(assemble(scenario))


def default_model_factory(problem: ThermalProblem) -> ThermalModel:
    """Pick the best available MODFLOW coupling for step-by-step runs.

    Driving libmf6 directly lets the inlet temperature change every step without restarting
    anything. If the library or modflowapi is missing, it falls back to re-running the mf6
    executable with restarts, which is slower but needs only the executable.
    """
    from importlib.util import find_spec

    from .mf6 import Mf6ApiSession, Mf6RestartSession, find_libmf6
    try:
        find_libmf6()
    except FileNotFoundError:
        return Mf6RestartSession(problem)
    return Mf6ApiSession(problem) if find_spec("modflowapi") else Mf6RestartSession(problem)


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened in one closed-loop step."""
    t_d: float
    T_in_C: float
    T_out_C: float
    heat_removed_W: float
    Q_load_W: float
    pump_power_W: float
    field: np.ndarray          # temperature of every cell after the step


class CoupledStepper:
    """One closed-loop step at a time: T_out -> controller -> T_in -> model.advance().

    Shared by run_coupled() (which loops it to the end) and the live GUI
    (which calls it from a timer), so both run exactly the same physics."""

    def __init__(self, model: ThermalModel, controller: InletController, T_max_C: float = 95.0):
        self.model, self.controller, self.T_max_C = model, controller, T_max_C
        self.m_dot = model.problem.scenario.operation.mass_flow_kg_s
        self.T_out_C = model.outlet_temperature()
        self.runaway = False

    @property
    def done(self) -> bool:
        """True once the run has reached its end time or been stopped by a runaway."""
        return self.model.finished or self.runaway

    def step(self) -> StepRecord | None:
        """Advance one closed-loop step; None if the run is over (end time or runaway).

        The controller looks at the water coming back from the mine and decides what temperature
        goes back in, the model advances one step with that inlet, and the heat removed is
        worked out from the temperature drop across the mine. If the returning water is hotter
        than T_max_C the mine can't keep up with the load, so the run is flagged as a runaway
        and stopped.
        """
        if self.model.finished:
            return None
        if self.T_out_C > self.T_max_C:
            self.runaway = True
            return None
        state = self.controller.inlet_state(self.T_out_C, self.model.time_d)
        self.model.set_inlet_temperature(state.T_in_C)
        self.model.advance()
        field = self.model.temperatures()
        self.T_out_C = float(field[self.model.problem.graph.outlet_cell])
        return StepRecord(t_d=self.model.time_d, T_in_C=state.T_in_C, T_out_C=self.T_out_C,
                          heat_removed_W=float(advected_heat_W(self.m_dot, state.T_in_C, self.T_out_C)),
                          Q_load_W=state.Q_load_W, pump_power_W=state.pump_power_W, field=field)


def run_coupled(scenario: Scenario, controller: InletController | None = None,
                model_factory: ModelFactory = default_model_factory,
                T_max_C: float = 95.0) -> RunResult:
    """Closed loop: every step, the controller turns T_out into T_in.

    Stops early (runaway=True) if the drift outlet passes T_max_C, meaning
    the mine cannot reject the load."""
    problem = assemble(scenario)
    controller = controller or ConstantInlet(scenario.operation.T_in_C)
    records: list[StepRecord] = []
    with model_factory(problem) as model:
        stepper = CoupledStepper(model, controller, T_max_C)
        while (rec := stepper.step()) is not None:
            records.append(rec)
    return result_from_records(problem, records, runaway=stepper.runaway)


def result_from_records(problem: ThermalProblem, records: list[StepRecord],
                        runaway: bool = False) -> RunResult:
    """Stack a list of per-step records into a single RunResult with one array per quantity."""
    col = lambda name: np.array([getattr(r, name) for r in records])
    return RunResult(problem=problem, times_d=col("t_d"), T_in_C=col("T_in_C"),
                     T_out_C=col("T_out_C"), heat_removed_W=col("heat_removed_W"),
                     temperatures=np.array([r.field for r in records]),
                     Q_load_W=col("Q_load_W"), pump_power_W=col("pump_power_W"), runaway=runaway)

"""A live run without any GUI: build once, then step on demand.

SimulationSession owns the problem, the engine (any ThermalModel from the
registry), the closed-loop stepper and the history. The GUI drives it from a
separate process (live/process.py); tests and notebooks can call it directly.

Building runs as a staged pipeline (live/progress.py) and reports each stage,
so a window can show a progress bar with the time left:

    s = SimulationSession(config, progress=lambda p: print(p.text, f"{p.fraction():.0%}"))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..assembly import ThermalProblem, assemble
from ..engines import ENGINES
from ..simulation import CoupledStepper
from .config import LiveConfig, build_controller, build_scenario
from .history import History
from .progress import GENERIC_STAGES, BuildPipeline, BuildProgress


@dataclass(frozen=True)
class BuildInfo:
    """What a finished build tells the window: the problem, which engine and device it used, and
    how long each stage took.

    `speed_factor` records how this machine compared with the built-in time estimates, so the
    next build's progress bar can start from a better guess.
    """
    problem: ThermalProblem
    engine_label: str
    build_seconds: float
    summary: str
    device_label: str = "CPU"
    solver_label: str = ""
    stage_seconds: dict = field(default_factory=dict)     # stage key -> seconds
    notices: tuple[str, ...] = ()                          # e.g. why the GPU wasn't used
    speed_factor: float = 1.0                              # how this machine compared with the estimates

    @property
    def compute_label(self) -> str:
        """Device and solver as one line, e.g. for the panel under the Build button."""
        return self.device_label + (f" · {self.solver_label}" if self.solver_label else "")


class SimulationSession:
    """One live run: built once from the settings, then stepped on demand.

    The build is split into named stages (settings, grid, properties, the engine's own stages,
    surface loop) so a progress bar can follow along. After that the session holds the model,
    the closed-loop stepper and the History of what happened, and `step()` advances as many
    steps as the caller asks for or has time for.
    """
    def __init__(self, config: LiveConfig, max_frames: int = 600,
                 progress: Callable[[BuildProgress], None] | None = None,
                 history=None, speed_factor: float = 1.0):
        t0 = time.perf_counter()
        info = ENGINES.get(config.engine)
        if info is None:
            raise ValueError(f"unknown engine '{config.engine}'")
        reason = info.check()
        if reason:
            raise RuntimeError(f"{info.label} is not available: {reason}")
        stages = GENERIC_STAGES[:3] + info.stages + GENERIC_STAGES[4:]
        wants_gpu = info.supports_device and config.device != "cpu"
        pipe = BuildPipeline(progress, stages, speed_factor=speed_factor, gpu=wants_gpu,
                             solver=config.solver, surface=config.surface, engine=config.engine)
        self.config = config
        with pipe.stage("settings"):
            self.scenario = build_scenario(config)
        with pipe.stage("grid"):
            graph = self.scenario.grid.build(self.scenario.drift)
        pipe.set_cells(graph.n_cells)
        with pipe.stage("properties"):
            self.problem = assemble(self.scenario, graph=graph)
        options = dict(progress=pipe.begin)
        if info.supports_device:
            options.update(device=config.device, solver=config.solver)
        self.model = info.factory(self.problem, **options)
        with pipe.stage("surface"):
            self.stepper = CoupledStepper(self.model, build_controller(config, self.scenario), config.T_max_C)
        self.history = history if history is not None else History(max_frames)
        self.history.clear(initial_field=self.model.temperatures())
        pipe.finish()

        device = getattr(self.model, "device", None)
        notices = []
        why = getattr(self.model, "fallback_reason", None)
        if why and config.device not in ("cpu",):
            notices.append(f"Running on the CPU: {why}")
        note = getattr(self.model, "selection_note", None)
        if note:
            notices.append(note)
        self.info = BuildInfo(problem=self.problem, engine_label=info.label,
                              build_seconds=time.perf_counter() - t0, summary=self.scenario.summary(),
                              device_label=device.label if device is not None else "CPU",
                              solver_label=getattr(self.model, "solver_description", ""),
                              stage_seconds=dict(pipe.seconds), notices=tuple(notices),
                              speed_factor=pipe.speed_factor)
        self.step_seconds = 0.0          # running average wall time per step
        self._notices_seen = len(getattr(self.model, "notices", []))

    @property
    def done(self) -> bool:
        """True once the run has ended, either at its end time or through a runaway."""
        return self.stepper.done

    @property
    def runaway(self) -> bool:
        """True if the run stopped because the loop water got too hot for the mine to cool."""
        return self.stepper.runaway

    @property
    def time_d(self) -> float:
        """Simulated time reached, in days."""
        return self.model.time_d

    @property
    def dt_days(self) -> float:
        """Length of one time step, in days."""
        return self.scenario.numerics.dt_days

    @property
    def sim_days(self) -> float:
        """Planned length of the run, in days."""
        return self.scenario.numerics.sim_days

    def step(self, n: int = 1, wall_budget_s: float | None = None) -> int:
        """Advance up to n steps (stopping early at the end, on runaway, or when
        the wall-clock budget is used up). Returns how many steps ran."""
        t0, done = time.perf_counter(), 0
        while done < n:
            rec = self.stepper.step()
            if rec is None:
                break
            self.history.append(rec)
            done += 1
            if wall_budget_s is not None and time.perf_counter() - t0 > wall_budget_s:
                break
        if done:
            per = (time.perf_counter() - t0) / done
            self.step_seconds = per if self.step_seconds == 0 else 0.8 * self.step_seconds + 0.2 * per
        return done

    def new_notices(self) -> list[str]:
        """Solver/device changes since the last call (e.g. a GPU failure mid-run)."""
        all_ = getattr(self.model, "notices", [])
        out = list(all_[self._notices_seen:])
        self._notices_seen = len(all_)
        return out

    @property
    def compute_label(self) -> str:
        """What is solving right now (it can change if the GPU fails mid-run)."""
        device = getattr(self.model, "device", None)
        solver = getattr(self.model, "solver_description", "")
        return (device.label if device is not None else "CPU") + (f" · {solver}" if solver else "")

    def close(self) -> None:
        """Release the model's resources."""
        self.model.close()

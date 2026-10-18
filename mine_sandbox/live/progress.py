"""Build progress: named stages, timings and an estimate of the time left.

Building a model is a fixed pipeline (like staged filter execution: each
stage hands its result to the next and reports when it starts). Each stage
has an expected cost that grows with the number of cells; the estimates are
rescaled by how fast the finished stages actually ran on this machine, so the
ETA corrects itself as the build goes.

    pipe = BuildPipeline(callback, stages=FV_STAGES)
    with pipe.stage("grid"):
        graph = ...
    pipe.set_cells(graph.n_cells, device_is_gpu=True)
    ...
    pipe.finish()      # pipe.seconds -> {"grid": 0.04, ...}

`callback` receives a BuildProgress (plain data: it crosses process boundaries).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

STAGE_LABELS = {
    "settings": "Reading settings",
    "grid": "Generating the grid",
    "properties": "Assigning rock properties",
    "flow": "Solving the water flow",
    "matrix": "Assembling the heat matrix",
    "solver": "Preparing the solver",
    "engine": "Writing and starting MODFLOW 6",
    "surface": "Starting the surface loop",
    "done": "Ready",
}
FV_STAGES = ("settings", "grid", "properties", "flow", "matrix", "solver", "surface")
GENERIC_STAGES = ("settings", "grid", "properties", "engine", "surface")


@dataclass(frozen=True)
class BuildProgress:
    """A snapshot of build progress, sent each time a stage starts.

    It's plain data so it can cross from the simulation process to the window. Because reports
    only arrive at stage boundaries, `fraction` and `eta_s` take the time since the report was
    sent, letting the progress bar keep moving smoothly in between.
    """
    stage: str               # stage key ("done" when finished)
    label: str
    index: int               # 0-based stage number
    count: int               # number of stages
    done_s: float            # seconds the finished stages took
    stage_s: float           # expected seconds of this stage
    total_s: float           # done_s + expected seconds of this and the remaining stages
    elapsed_s: float         # wall time since the build started, when this was sent
    stage_elapsed_s: float = 0.0   # wall time already spent in this stage, when this was sent
    finished: bool = False

    def fraction(self, since_sent_s: float = 0.0) -> float:
        """Estimated share of the build done, `since_sent_s` seconds after this report.

        Progress within a stage is capped just short of its end, so the bar never claims a stage
        is finished before it actually is.
        """
        if self.finished:
            return 1.0
        if self.total_s <= 0:
            return 0.0
        inside = min(self.stage_elapsed_s + since_sent_s, 0.95 * self.stage_s)   # never claim a stage is over
        return max(0.0, min(0.99, (self.done_s + inside) / self.total_s))

    def eta_s(self, since_sent_s: float = 0.0) -> float | None:
        """Estimated seconds left, or None once the current stage runs past its estimate."""
        if self.finished:
            return 0.0
        in_stage = self.stage_elapsed_s + since_sent_s
        left_here = self.stage_s - in_stage
        if left_here < -0.25 * max(self.stage_s, 0.4):
            return None
        return max(0.0, left_here) + max(0.0, self.total_s - self.done_s - self.stage_s)

    @property
    def text(self) -> str:
        """Short status line, e.g. 'Solving the water flow (4/7)'."""
        if self.finished:
            return "Ready"
        return f"{self.label} ({self.index + 1}/{self.count})"


def expected_stage_seconds(stage: str, n_cells: int | None, *, gpu: bool = False, solver: str = "auto",
                           surface: str = "fixed_inlet", engine: str = "fv") -> float:
    """Rough cost of one stage on a typical desktop CPU (seconds); rescaled while building."""
    n = n_cells if n_cells else 50_000
    if stage == "settings":
        return 0.02
    if stage == "grid":
        return 0.02 + 4e-7 * n
    if stage == "properties":
        return 0.02 + 2.5e-6 * n
    if stage == "flow":
        return 0.01 + 1e-7 * n
    if stage == "matrix":
        return 0.02 + 6e-7 * n
    if stage == "solver":
        if solver == "iterative":
            return 0.05 + 2.2e-5 * n
        if solver == "direct" or (solver == "auto" and n <= 3_000 and not gpu):
            return 0.02 + 1e-5 * n
        base = 0.02 + 6e-7 * n            # block solver: small water LU + four trial solves
        return base + (1.5 if gpu else 0.0)   # first-time kernel compilation on the GPU
    if stage == "engine":                 # MODFLOW 6: write the input files, load libmf6
        return 1.0 + 3e-5 * n
    if stage == "surface":
        return 1.2 if surface == "fixed_load" else 0.02
    return 0.05


class BuildPipeline:
    """Runs the build stages in order and reports a BuildProgress at each stage start.

    Each stage has an expected cost from `expected_stage_seconds`. As stages finish, the
    pipeline compares how long they really took against those estimates and rescales the rest,
    so the time-left estimate gets better as the build goes on.
    """

    def __init__(self, callback: Callable[[BuildProgress], None] | None = None,
                 stages: tuple[str, ...] = FV_STAGES, *, speed_factor: float = 1.0, **cost_kw):
        self.callback = callback
        self.stages = tuple(stages)
        self.seconds: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._cells: int | None = None
        self._cost_kw = cost_kw
        self._scale = max(0.1, float(speed_factor))   # learnt from earlier builds on this machine
        self._current: str | None = None
        self._stage_t0 = self._t0
        self._finished = False

    # estimates
    def set_cells(self, n_cells: int, **cost_kw) -> None:
        """Tell the pipeline how many cells the model has, now that the grid exists, so later stage
        estimates can scale with it.
        """
        self._cells = int(n_cells)
        self._cost_kw.update(cost_kw)

    def _expected(self, stage: str) -> float:
        """Expected seconds for a stage on this machine (the raw estimate times the learnt speed
        factor).
        """
        return self._scale * expected_stage_seconds(stage, self._cells, **self._cost_kw)

    def _observe(self, stage: str, actual: float) -> None:
        """Rescale future estimates by how this machine did on a stage big enough to time.

        Each observation is blended with the old factor and clamped, so one unusually slow or
        fast stage can't throw off the estimate.
        """
        raw = expected_stage_seconds(stage, self._cells, **self._cost_kw)
        if raw >= 0.05 and actual > 0:
            ratio = min(5.0, max(0.2, actual / raw))
            self._scale = 0.5 * self._scale + 0.5 * ratio

    @property
    def speed_factor(self) -> float:
        """How much slower (>1) or faster (<1) this machine is than the built-in estimates."""
        return self._scale

    # stages
    def begin(self, stage: str) -> None:
        """Start a stage (ending the current one)."""
        now = time.perf_counter()
        self._end_current(now)
        self._current, self._stage_t0 = stage, now
        self._emit(stage, now)

    def _end_current(self, now: float) -> None:
        """Close the running stage, record how long it took and learn from it."""
        if self._current is not None:
            took = now - self._stage_t0
            self.seconds[self._current] = self.seconds.get(self._current, 0.0) + took
            self._observe(self._current, took)
            self._current = None

    @contextmanager
    def stage(self, key: str):
        """Context manager that runs the body as a named stage."""
        self.begin(key)
        yield
        self._end_current(time.perf_counter())

    def finish(self) -> None:
        """Close the last stage and report that the build is done."""
        now = time.perf_counter()
        self._end_current(now)
        self._finished = True
        self._emit("done", now)

    @property
    def elapsed_s(self) -> float:
        """Wall time since the build started."""
        return time.perf_counter() - self._t0

    def _emit(self, stage: str, now: float) -> None:
        """Send a progress report: finished stages count at their real time, the rest at their
        estimates.
        """
        if self.callback is None:
            return
        finished = stage == "done"
        idx = len(self.stages) if finished else (self.stages.index(stage) if stage in self.stages else 0)
        exp = [self._expected(s) for s in self.stages]
        done = now - self._t0                    # finished stages count at what they really took
        left = sum(exp[idx:])
        self.callback(BuildProgress(
            stage=stage, label=STAGE_LABELS.get(stage, stage), index=min(idx, len(self.stages) - 1),
            count=len(self.stages), done_s=done, stage_s=0.0 if finished else exp[idx],
            total_s=done + left, elapsed_s=done, stage_elapsed_s=0.0, finished=finished))

"""Drives a SimulationSession for the window. Three interchangeable workers,
all with the same slots and signals, so MainWindow doesn't care which runs:

    ProcessWorker     the default. The session lives in its own process
                      (live/process.py); this object sits on the GUI thread and
                      only polls for results, so the window never waits for the
                      solver. Builds can be cancelled (the process is replaced).
    SimulationWorker  the session in this process: moved to a QThread
                      (backend="thread") or called synchronously (backend="inline",
                      used by tests and --screenshot).

Signals
    built(BuildInfo, History)   failed(str)            busy(str)
    ticked(TickInfo)            progress(BuildProgress)
    devicesFound(DeviceReport)  notice(str)            runFailed(str)
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..live import History, LiveConfig, SimulationProcess, SimulationSession, TickInfo
from ..live.process import RateMeter

__all__ = ["ProcessWorker", "SimulationWorker", "TickInfo"]


class _Signals(QObject):
    """The signals every worker emits, so the main window can connect to any of them the same way."""
    built = Signal(object, object)      # BuildInfo, History
    failed = Signal(str)                # the build failed
    busy = Signal(str)                  # status text while building
    ticked = Signal(object)             # TickInfo
    progress = Signal(object)           # BuildProgress
    devicesFound = Signal(object)       # DeviceReport
    notice = Signal(str)                # e.g. the GPU failed and the run carries on on the CPU
    runFailed = Signal(str)             # an error while stepping (the run stops)



class ProcessWorker(_Signals):
    """Runs the simulation in a separate process and relays its events as Qt signals.

    A timer on the GUI thread polls the process every few milliseconds, so nothing here ever
    blocks the window. It also watches over the process: if it dies before starting it gives up
    with advice, if it dies while checking for a GPU it restarts on the CPU only, and if it dies
    mid-run it restarts and tells the user. Cancelling a build simply replaces the process.
    """
    POLL_MS = 15
    WATCH_S = 0.5                       # how often to check the process is still alive

    def __init__(self, max_frames: int = 600):
        super().__init__()
        self.max_frames = max_frames
        self.speed_days_per_s = 10.0
        self.speed_factor = 1.0          # build-time estimates, learnt from earlier builds
        self.history: History | None = None
        self.building = False
        self.devices = None
        self._sid = 0
        self._last_watch = 0.0
        self._hello = False                # this process has started up
        self._devices_seen = False         # ... and finished checking for a GPU
        self._force_cpu = False            # set if the GPU check ever crashed the process
        self._given_up = False
        self._proc = SimulationProcess()
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # the window's MainWindow calls start() for thread workers; nothing to do here
    @Slot()
    def start(self) -> None:
        pass

    # commands
    @Slot(object)
    def build(self, config: LiveConfig) -> None:
        """Start building a model from the settings, replacing any build still in progress."""
        if self.building:                # a new build replaces one that is still running
            self._restart()
        self.building = True
        self.busy.emit("Building model…" if self.devices else "Starting the simulation process…")
        self._sid = self._proc.build(config, self.max_frames, self.speed_factor)

    @Slot()
    def cancel_build(self) -> None:
        """Abandon the current build by restarting the process."""
        if self.building:
            self._restart()
            self.building = False
            self.failed.emit("build cancelled")

    @Slot()
    def play(self) -> None:
        """Start or resume the run."""
        self._proc.play()

    @Slot()
    def pause(self) -> None:
        """Pause the run."""
        self._proc.pause()

    @Slot()
    def step_once(self) -> None:
        """Run exactly one step."""
        self._proc.step(1)

    @Slot(float)
    def set_speed(self, days_per_s: float) -> None:
        """Set the playback speed in simulated days per second."""
        self.speed_days_per_s = float(days_per_s)
        self._proc.set_speed(self.speed_days_per_s)

    @Slot()
    def shutdown(self) -> None:
        """Stop polling and close the simulation process cleanly."""
        self._timer.stop()
        self._proc.close()

    # events
    def _poll(self) -> None:
        """Turn waiting process events into signals, then check the process is still alive.

        Events from a replaced session are ignored. Only the latest tick of a poll is emitted,
        since the window only needs to know where the run is now.
        """
        if self._given_up:
            return
        last_tick = None
        for ev in self._proc.poll():
            kind = ev[0]
            if kind == "hello":
                self._hello = True
                continue
            if kind == "devices":
                self._devices_seen = True
                self.devices = ev[1]
                self.devicesFound.emit(ev[1])
                continue
            sid = ev[1]
            if sid not in (self._sid, 0):
                continue                                  # from a session that has been replaced
            if kind == "progress":
                self.progress.emit(ev[2])
            elif kind == "built":
                info, initial = ev[2], ev[3]
                self.building = False
                self.speed_factor = info.speed_factor
                self.history = History(self.max_frames)
                self.history.clear(initial_field=initial)
                self.built.emit(info, self.history)
            elif kind == "failed":
                self.building = False
                self.failed.emit(ev[2])
            elif kind == "batch":
                if self.history is not None:
                    self.history.extend(ev[2])
                last_tick = ev[3]
            elif kind == "notice":
                self.notice.emit(ev[2])
            elif kind == "run_failed":
                self.runFailed.emit(ev[2])
        if last_tick is not None:
            self.ticked.emit(last_tick)
        now = time.perf_counter()
        if now - self._last_watch > self.WATCH_S:
            self._last_watch = now
            if not self._proc.alive:
                code = self._proc.exitcode
                was_building = self.building
                self.building = False
                if not self._hello:
                    # it died before starting up: restarting would just fail again
                    self._given_up = True
                    self._timer.stop()
                    self._proc.kill()
                    self.failed.emit(
                        f"the simulation process could not start (exit code {code}). If you opened the "
                        "window from your own script, put that code under  if __name__ == '__main__':  "
                        "— or run with  --backend thread")
                    return
                if not self._devices_seen and not self._force_cpu:
                    # it died while checking for a GPU: carry on without one
                    self._force_cpu = True
                    self._restart()
                    self.notice.emit(f"Checking for a GPU crashed the simulation process (exit code {code}); "
                                     "it has been restarted on the CPU only. Check the NVIDIA driver and "
                                     "CuPy install to use the GPU.")
                    return
                self._restart()
                msg = (f"the simulation process stopped unexpectedly (exit code {code}); "
                       "it has been restarted — press Build model to continue"
                       + ("" if self._force_cpu else ". If this keeps happening on the GPU, set Compute to CPU"))
                (self.failed if was_building else self.runFailed).emit(msg)

    def _restart(self) -> None:
        """Replace the simulation process with a fresh one, keeping the speed and CPU-only setting."""
        self._proc.kill()
        self._hello = self._devices_seen = False
        self._proc = SimulationProcess(detect_devices=True, force_cpu=self._force_cpu)
        self._proc.set_speed(self.speed_days_per_s)
        self._sid = 0

    @property
    def pid(self) -> int | None:
        """Operating-system id of the current simulation process."""
        return self._proc.pid



class SimulationWorker(_Signals):
    """The session in this process. Each timer tick runs as many steps as the
    chosen speed calls for, within a wall-clock budget, then emits `ticked`."""
    TICK_MS = 40
    WALL_BUDGET_S = 0.025               # max compute per tick: leaves the GUI thread time to draw

    def __init__(self, max_frames: int = 600):
        super().__init__()
        self.max_frames = max_frames
        self.session: SimulationSession | None = None
        self.playing = False
        self.building = False
        self.speed_days_per_s = 10.0
        self.speed_factor = 1.0
        self.devices = None
        self._timer: QTimer | None = None
        self._last = time.perf_counter()
        self._carry = 0.0
        self._rate = RateMeter()

    # lifecycle
    @Slot()
    def start(self) -> None:
        """Called in the worker thread once it starts."""
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @Slot()
    def shutdown(self) -> None:
        """Stop ticking and close the session."""
        if self._timer:
            self._timer.stop()
        self._close_session()

    def _close_session(self):
        """Close the current session, if any."""
        if self.session is not None:
            self.session.close()
            self.session = None

    # commands
    @Slot(object)
    def build(self, config: LiveConfig) -> None:
        """Build a session right here (blocking this thread) and report the result."""
        self.playing = False
        self._close_session()
        self.busy.emit("Building model…")
        self.building = True
        try:
            self.session = SimulationSession(config, self.max_frames, progress=self.progress.emit,
                                             speed_factor=self.speed_factor)
        except Exception as e:                      # report any setup error to the user
            self.building = False
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.building = False
        self.speed_factor = self.session.info.speed_factor
        self.built.emit(self.session.info, self.session.history)
        self._emit()

    @Slot()
    def cancel_build(self) -> None:
        """Does nothing: an in-process build can't be interrupted."""
        pass                                         # in-process builds can't be interrupted

    @Slot()
    def play(self) -> None:
        """Start or resume the run."""
        if self.session and not self.session.done:
            self.playing = True
            self._last = time.perf_counter()
            self._carry = 0.0
            self._rate.clear()
            self._rate.add(self.session.time_d)
            self._emit()

    @Slot()
    def pause(self) -> None:
        """Pause the run."""
        self.playing = False
        self._emit()

    @Slot()
    def step_once(self) -> None:
        """Run exactly one step."""
        if self.session:
            self.playing = False
            self._run(1)
            self._emit()

    @Slot(float)
    def set_speed(self, days_per_s: float) -> None:
        """Set the playback speed in simulated days per second."""
        self.speed_days_per_s = float(days_per_s)

    # ticking
    @Slot()
    def _tick(self) -> None:
        """Run however many steps the chosen speed calls for since the last tick, within the
        wall-clock budget.

        If the solver can't keep up, the shortfall is dropped rather than carried over, so the
        run never tries to catch up in a burst.
        """
        now = time.perf_counter()
        elapsed, self._last = min(now - self._last, 0.25), now
        s = self.session
        if not (self.playing and s):
            return
        if math.isinf(self.speed_days_per_s):
            n = 1_000_000
        else:
            self._carry += elapsed * self.speed_days_per_s / s.dt_days
            n = int(self._carry)
        if n == 0:
            return
        ran = self._run(n, wall_budget_s=self.WALL_BUDGET_S)
        self._rate.add(s.time_d)
        if math.isinf(self.speed_days_per_s):
            self._carry = 0.0
        else:
            self._carry = 0.0 if ran < n else self._carry - ran     # can't keep up: don't pile up debt
        if s.done:
            self.playing = False
        self._emit()

    def _run(self, n: int, wall_budget_s: float | None = None) -> int:
        """Step the session, turning any error into a runFailed signal and passing on solver notices."""
        try:
            ran = self.session.step(n, wall_budget_s=wall_budget_s)
        except Exception as e:  # noqa: BLE001
            self.playing = False
            self.runFailed.emit(f"{type(e).__name__}: {e}")
            return 0
        for text in self.session.new_notices():
            self.notice.emit(text)
        return ran

    def _emit(self) -> None:
        """Tell the window where the run is now."""
        s = self.session
        if s is None:
            return
        self.ticked.emit(TickInfo(t_d=s.time_d, n_steps=s.history.n_steps, playing=self.playing,
                                  done=s.done, runaway=s.runaway, step_ms=1000 * s.step_seconds,
                                  sim_days=s.sim_days, days_per_s=self._rate.days_per_s if self.playing else 0.0,
                                  compute_label=s.compute_label))

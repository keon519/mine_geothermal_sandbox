"""Run the simulation in its own process, so the window never waits for it.

A thread is not enough in Python: the solver holds the interpreter lock for
most of each step, so a simulation thread and the GUI thread take turns and
both slow down. A separate process gets its own interpreter (and its own CPU
core, or the GPU), and the window only reads results.

    GUI process                                   simulation process
    -----------                                   ------------------
    SimulationProcess.build(config)  --cmd-->     SimulationSession (staged build)
                                     <--evt--     ("progress", BuildProgress) per stage
                                     <--evt--     ("built", BuildInfo, initial field)
    .play() / .pause() / .step()     --cmd-->     paced stepping at the chosen speed
                                     <--evt--     ("batch", StepBatch, TickInfo) ~30x a second
    History.extend(batch)                         (only the fields the History keeps are sent)

Commands and events are plain picklable tuples on two multiprocessing queues.
The worker says ("hello", pid) first, then ("devices", DeviceReport), so the
window can tell a process that failed to start from one whose GPU check crashed.
The worker paces itself (simulated days per wall second, or as fast as
possible) and never waits for the GUI: if the window falls behind, batches
are merged rather than queued (the client acknowledges what it has read).

Uses the "spawn" start method on every platform: it is the only one on
Windows, and it is the safe one with CUDA. Nothing here imports Qt.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import signal
import time
import traceback
from collections import deque
from dataclasses import dataclass

from .history import BatchRecorder

CTX = mp.get_context("spawn")
PUBLISH_S = 0.033          # at most ~30 batches a second
CHUNK_S = 0.06             # longest stretch of stepping between looks at the command queue
MAX_IN_FLIGHT = 3          # unacknowledged batches before the worker starts merging them
MAX_SPEED = math.inf       # "as fast as possible"


@dataclass(frozen=True)
class TickInfo:
    """Where the run is, sent with every batch."""
    t_d: float
    n_steps: int
    playing: bool
    done: bool
    runaway: bool
    step_ms: float                 # wall time per closed-loop step (solver + surface loop)
    sim_days: float = 0.0
    days_per_s: float = 0.0        # simulated days per wall second actually achieved lately
    compute_label: str = ""
    session_id: int = 0


class RateMeter:
    """Simulated days per wall second over the last `window_s` seconds."""

    def __init__(self, window_s: float = 1.5):
        self.window_s = window_s
        self._samples = deque(maxlen=256)           # (wall time, simulated day)

    def clear(self) -> None:
        """Forget past samples (e.g. after pausing, so the rate isn't dragged down by the pause)."""
        self._samples.clear()

    def add(self, t_d: float, now: float | None = None) -> None:
        """Record the simulated day reached at a wall-clock moment."""
        self._samples.append((time.perf_counter() if now is None else now, t_d))

    @property
    def days_per_s(self) -> float:
        """Simulation speed over the recent window, in simulated days per wall second."""
        if len(self._samples) < 2:
            return 0.0
        now, t_now = self._samples[-1]
        old = next((smp for smp in self._samples if now - smp[0] <= self.window_s), self._samples[0])
        if old is self._samples[-1]:
            old = self._samples[-2]
        dt = now - old[0]
        return (t_now - old[1]) / dt if dt > 0 else 0.0



# client (GUI side)

class SimulationProcess:
    """The window's handle on the simulation process.

    Starting it launches a separate Python process running `worker_main`. The window sends it
    short commands (build, play, pause, step, change speed) and collects events with `poll()`.
    Each build gets a new session id, so events from an older build that arrive late can be
    recognised and ignored.
    """
    def __init__(self, detect_devices: bool = True, force_cpu: bool = False):
        self._cmd = CTX.Queue()
        self._evt = CTX.Queue()
        self.force_cpu = force_cpu
        self._proc = CTX.Process(target=worker_main, args=(self._cmd, self._evt, detect_devices, force_cpu),
                                 name="mine-sandbox-simulation", daemon=True)
        self._proc.start()
        self.session_id = 0
        self._closed = False

    # commands
    def build(self, config, max_frames: int = 600, speed_factor: float = 1.0) -> int:
        """Ask the worker to build a new session from the settings; returns the session id its
        events will carry.
        """
        self.session_id += 1
        self._put("build", self.session_id, config, max_frames, speed_factor)
        return self.session_id

    def play(self) -> None:
        """Start or resume stepping at the chosen speed."""
        self._put("play")

    def pause(self) -> None:
        """Stop stepping."""
        self._put("pause")

    def step(self, n: int = 1) -> None:
        """Pause and run exactly `n` steps."""
        self._put("step", int(n))

    def set_speed(self, days_per_s: float) -> None:
        """Set the target speed in simulated days per second (MAX_SPEED for as fast as possible)."""
        self._put("speed", float(days_per_s))

    def _put(self, *cmd) -> None:
        """Queue a command, unless the process has been closed."""
        if not self._closed:
            self._cmd.put(cmd)

    # events
    def poll(self, max_events: int = 500, timeout: float = 0.0) -> list[tuple]:
        """Events that have arrived (non-blocking unless timeout > 0).

        Reading batches also acknowledges them, which tells the worker the window has caught up
        and it can send more.
        """
        out, batches = [], 0
        try:
            ev = self._evt.get(timeout=timeout) if timeout > 0 else self._evt.get_nowait()
            while True:
                out.append(ev)
                batches += ev[0] == "batch"
                if len(out) >= max_events:
                    break
                ev = self._evt.get_nowait()
        except queue.Empty:
            pass
        except (EOFError, OSError):          # the process died mid-message
            pass
        if batches:
            self._put("ack", batches)
        return out

    # lifecycle
    @property
    def alive(self) -> bool:
        """True while the worker process is running."""
        return self._proc.is_alive()

    @property
    def exitcode(self) -> int | None:
        """The worker's exit code once it has stopped (negative if killed by a signal)."""
        return self._proc.exitcode

    @property
    def pid(self) -> int | None:
        """Operating-system id of the worker process."""
        return self._proc.pid

    def close(self, timeout: float = 3.0) -> None:
        """Ask the worker to finish (closing MODFLOW cleanly), then make sure it is gone."""
        if self._closed:
            return
        if self._proc.is_alive():
            try:
                self._cmd.put(("shutdown",))
            except (OSError, ValueError):
                pass
            self._proc.join(timeout)
        self._closed = True
        self.kill()

    def kill(self) -> None:
        """Stop at once (cancelling a build that is in progress)."""
        self._closed = True
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
        for q in (self._cmd, self._evt):
            q.cancel_join_thread()
            q.close()



# worker (simulation process)

def worker_main(cmd_q, evt_q, detect_devices: bool = True, force_cpu: bool = False) -> None:
    """Entry point of the simulation process.

    It ignores Ctrl+C (the window handles that), optionally hides the GPU, announces itself,
    reports what compute devices it found, and then hands over to the command loop. Announcing
    itself before the GPU check lets the window tell a process that never started apart from one
    that crashed while probing the GPU.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)         # Ctrl+C belongs to the window
    except (ValueError, AttributeError):
        pass
    import os
    if force_cpu:                                            # e.g. the GPU check crashed last time
        os.environ["MINE_SANDBOX_DEVICE"] = "cpu"
    evt_q.put(("hello", os.getpid()))
    if detect_devices:
        from ..compute import detect_devices as detect
        try:
            evt_q.put(("devices", detect()))
        except Exception as e:  # noqa: BLE001
            evt_q.put(("notice", 0, f"Could not check for a GPU: {type(e).__name__}: {e}"))
    _Worker(cmd_q, evt_q).run()


class _Worker:
    """The command loop inside the simulation process.

    It alternates between handling commands, stepping the session at the requested pace, and
    publishing batches of results. Pacing works by accumulating fractional steps owed from
    elapsed wall time, so a slow speed still advances smoothly. Flow control keeps at most a few
    unacknowledged batches in flight; beyond that, new steps pile up in the recorder and go out
    together once the window catches up.
    """
    def __init__(self, cmd_q, evt_q):
        self.cmd_q, self.evt_q = cmd_q, evt_q
        self.session = None
        self.sid = 0
        self.recorder: BatchRecorder | None = None
        self.playing = False
        self.speed = 10.0
        self._carry = 0.0
        self._last = time.perf_counter()
        self._last_publish = 0.0
        self._in_flight = 0
        self._dirty = False
        self._rate = RateMeter()

    # loop
    def run(self) -> None:
        """Main loop: wait briefly for commands, handle them all, advance the simulation, then
        maybe publish.
        """
        while True:
            try:
                cmd = self.cmd_q.get(timeout=self._idle_timeout())
            except queue.Empty:
                cmd = None
            except (EOFError, OSError):
                return                                # the window has gone
            while cmd is not None:
                if cmd[0] == "shutdown":
                    self._close_session()
                    return
                self._handle(cmd)
                try:
                    cmd = self.cmd_q.get_nowait()
                except queue.Empty:
                    cmd = None
            try:
                self._advance()
            except Exception as e:  # noqa: BLE001 - report it; never let the worker die silently
                self.playing = False
                self._dirty = True
                self._emit("run_failed", self.sid, f"{type(e).__name__}: {e}", traceback.format_exc())
            self._maybe_publish()

    def _idle_timeout(self) -> float:
        """How long to wait for a command before doing something else: short when there's work due,
        longer when idle.
        """
        if self.recorder is not None and (self._dirty or self.recorder.pending) and self._in_flight >= MAX_IN_FLIGHT:
            return 0.01                               # waiting for the window to catch up
        if not (self.playing and self.session):
            return 0.1 if not self._dirty else 0.0
        if math.isinf(self.speed):
            return 0.0
        due = (1.0 - self._carry) * self.session.dt_days / max(self.speed, 1e-9)
        return min(max(due, 0.0), 0.02)

    # commands
    def _handle(self, cmd) -> None:
        """Act on one command from the window."""
        kind = cmd[0]
        if kind == "build":
            _, sid, config, max_frames, speed_factor = cmd
            self._build(sid, config, max_frames, speed_factor)
        elif kind == "play":
            if self.session and not self.session.done:
                self.playing = True
                self._last = time.perf_counter()
                self._carry = 0.0
                self._rate.clear()
            self._dirty = True
        elif kind == "pause":
            self.playing = False
            self._dirty = True
        elif kind == "step":
            if self.session:
                self.playing = False
                self.session.step(cmd[1])
                self._dirty = True
        elif kind == "speed":
            self.speed = cmd[1]
        elif kind == "ack":
            self._in_flight = max(0, self._in_flight - cmd[1])

    def _build(self, sid, config, max_frames, speed_factor) -> None:
        """Replace the current session with a new one, streaming build progress and reporting
        success or the error.
        """
        from .session import SimulationSession
        self._close_session()
        self.playing = False
        self.sid = sid
        self.recorder = BatchRecorder(max_frames, sid)
        self._in_flight = 0
        try:
            self.session = SimulationSession(
                config, max_frames, history=self.recorder, speed_factor=speed_factor,
                progress=lambda p: self._emit("progress", sid, p))
        except Exception as e:  # noqa: BLE001 - any setup error goes to the window
            self.session = None
            self._emit("failed", sid, f"{type(e).__name__}: {e}")
            return
        initial = self.recorder.latest()
        self.recorder.drain()
        self._emit("built", sid, self.session.info, initial)
        self._dirty = True

    def _close_session(self) -> None:
        """Close the current session, if any."""
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    # stepping
    def _advance(self) -> None:
        """Run however many steps are due at the current speed.

        If the solver can't keep up with the requested speed, the shortfall is dropped rather
        than carried forward, so the run doesn't try to sprint to catch up later.
        """
        s = self.session
        now = time.perf_counter()
        elapsed, self._last = min(now - self._last, 0.25), now
        if not (self.playing and s):
            return
        if math.isinf(self.speed):
            n = 1_000_000
        else:
            self._carry += elapsed * self.speed / s.dt_days
            n = int(self._carry)
        if n > 0:
            ran = s.step(n, wall_budget_s=CHUNK_S)
            if not math.isinf(self.speed):
                self._carry = 0.0 if ran < n else self._carry - ran      # can't keep up: no debt
            self._rate.add(s.time_d)
        if s.done:
            self.playing = False
            self._dirty = True
        for text in s.new_notices():
            self._emit("notice", self.sid, text)

    #publishing
    def _maybe_publish(self) -> None:
        """Send a batch of new results if there's something new, the window isn't too far behind,
        and it's been long enough since the last one.
        """
        if self.recorder is None or self.session is None:
            return
        if not (self._dirty or self.recorder.pending):
            return
        if self._in_flight >= MAX_IN_FLIGHT:
            return
        now = time.perf_counter()
        if self.playing and not self._dirty and now - self._last_publish < PUBLISH_S:
            return
        s = self.session
        batch = self.recorder.drain(with_latest=True)
        if batch.n_steps == 0:
            batch.latest = None
        tick = TickInfo(t_d=s.time_d, n_steps=self.recorder.n_steps, playing=self.playing, done=s.done,
                        runaway=s.runaway, step_ms=1000 * s.step_seconds, sim_days=s.sim_days,
                        days_per_s=self._rate.days_per_s, compute_label=s.compute_label, session_id=self.sid)
        self._emit("batch", self.sid, batch, tick)
        self._in_flight += 1
        self._last_publish = now
        self._dirty = False

    def _emit(self, *event) -> None:
        """Put an event on the queue to the window."""
        self.evt_q.put(event)

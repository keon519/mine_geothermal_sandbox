"""Everything a live run has produced, kept so the user can scrub back in time.

Scalars (T_in, T_out, heat, ...) are kept for every step. Full temperature
fields are kept as float32 "frames"; when there are more than `max_frames`
the older ones are thinned to every other frame and the recording stride
doubles, so memory stays bounded however long the run.

Which steps become frames is decided by FrameSchedule alone. The simulation
process runs its own copy of the schedule, so it only ships the fields that
the window's History will keep (see StepBatch / History.extend): the two
sides cannot disagree.

Thread-safe: the simulation thread (or the batch reader) appends, the GUI reads.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from ..simulation import StepRecord

SCALARS = ("t_d", "T_in_C", "T_out_C", "heat_removed_W", "Q_load_W", "pump_power_W")


class FrameSchedule:
    """Decides which steps are stored as full temperature frames. Step numbers start at 1; frame 0
    is the initial field.

    It starts by keeping every step. Each time the frame count passes `max_frames`, it asks for
    the stored frames to be halved (every other one dropped) and doubles the stride, so from
    then on only every second, fourth, eighth... step is kept. Early and late parts of the run
    end up evenly covered, and memory never grows past the cap.
    """

    def __init__(self, max_frames: int = 600):
        self.max_frames = max(2, int(max_frames))
        self.reset()

    def reset(self, has_initial: bool = True) -> None:
        """Start a new run, counting the initial field as frame 0 if there is one."""
        self.n = 0
        self.stride = 1
        self.count = 1 if has_initial else 0

    def next_step(self) -> tuple[bool, bool]:
        """Advance one step. Returns (store this step as a frame, then thin the frames)."""
        self.n += 1
        if self.n % self.stride:
            return False, False
        self.count += 1
        if self.count > self.max_frames:
            self.count = (self.count + 1) // 2          # frames[::2] keeps ceil(count / 2)
            self.stride *= 2
            return True, True
        return True, False


@dataclass
class StepBatch:
    """Steps produced since the last batch, as shipped from the simulation process."""
    session_id: int
    scalars: dict[str, np.ndarray]                      # one value per step
    frames: list[tuple[float, np.ndarray]] = field(default_factory=list)   # fields of the steps that are frames
    latest: np.ndarray | None = None                    # field after the last step (float32)

    @property
    def n_steps(self) -> int:
        """Number of steps in the batch."""
        return len(self.scalars["t_d"]) if self.scalars else 0


class _Growable:
    """A float64 column with amortised O(1) appends and O(1) snapshot reads."""

    def __init__(self, capacity: int = 1024):
        self._a = np.empty(capacity)
        self._n = 0

    def append(self, v: float) -> None:
        """Add one value, doubling the buffer when it's full."""
        if self._n == len(self._a):
            self._a = np.concatenate([self._a, np.empty(len(self._a))])
        self._a[self._n] = v
        self._n += 1

    def extend(self, v: np.ndarray) -> None:
        """Add many values at once, growing the buffer if needed."""
        v = np.asarray(v, float)
        need = self._n + len(v)
        if need > len(self._a):
            self._a = np.concatenate([self._a[:self._n], np.empty(max(need, 2 * len(self._a)) - self._n)])
        self._a[self._n:need] = v
        self._n = need

    def view(self) -> np.ndarray:
        """A copy of the values so far, safe to hand to another thread."""
        return self._a[:self._n].copy()

    def __len__(self) -> int:
        return self._n


class History:
    """Everything a live run has produced, kept so the window can replay any moment.

    Scalars are kept for every step, while full temperature fields are thinned by a
    FrameSchedule. All reads and writes take a lock, since results arrive on one thread while
    the GUI reads on another.
    """
    def __init__(self, max_frames: int = 600):
        self.max_frames = max_frames
        self._lock = threading.Lock()
        self._schedule = FrameSchedule(max_frames)
        self.clear()

    def clear(self, initial_field: np.ndarray | None = None) -> None:
        """Forget everything and start again, optionally with the starting field as frame 0."""
        with self._lock:
            self._scalars = {k: _Growable() for k in SCALARS}
            self._frame_t: list[float] = []
            self._frames: list[np.ndarray] = []
            self._schedule.reset(has_initial=initial_field is not None)
            self._latest = None
            if initial_field is not None:
                self._frame_t.append(0.0)
                self._frames.append(np.asarray(initial_field, np.float32))
                self._latest = self._frames[0]

    #  writing
    def append(self, rec: StepRecord) -> None:
        """Record one step from a StepRecord (used when the session runs in-process)."""
        with self._lock:
            for k in SCALARS:
                self._scalars[k].append(getattr(rec, k))
            field = np.asarray(rec.field, np.float32)
            self._latest = field
            self._record(rec.t_d, field)

    def extend(self, batch: StepBatch) -> None:
        """Add the steps of a StepBatch (fields only for the steps that are frames)."""
        n = batch.n_steps
        if n == 0:
            return
        with self._lock:
            for k in SCALARS:
                self._scalars[k].extend(batch.scalars[k])
            frames = iter(batch.frames)
            for _ in range(n):
                keep, thin = self._schedule.next_step()
                if keep:
                    t, f = next(frames)
                    self._frame_t.append(float(t))
                    self._frames.append(np.asarray(f, np.float32))
                    if thin:
                        self._thin()
            if next(frames, None) is not None:
                raise ValueError("batch carries more frames than the schedule asked for")
            if batch.latest is not None:
                self._latest = np.asarray(batch.latest, np.float32)

    def _record(self, t: float, field: np.ndarray) -> None:
        """Store a field as a frame if the schedule wants it, thinning older frames when told to."""
        keep, thin = self._schedule.next_step()
        if keep:
            self._frame_t.append(t)
            self._frames.append(field)
            if thin:
                self._thin()

    def _thin(self) -> None:
        """Drop every other stored frame."""
        self._frame_t = self._frame_t[::2]
        self._frames = self._frames[::2]

    # reading
    @property
    def n_steps(self) -> int:
        """Number of steps recorded."""
        return self._schedule.n

    @property
    def n_frames(self) -> int:
        """Number of full temperature frames currently stored."""
        with self._lock:
            return len(self._frames)

    def scalars(self) -> dict[str, np.ndarray]:
        """Copies of every scalar series (one value per step), keyed by name."""
        with self._lock:
            return {k: v.view() for k, v in self._scalars.items()}

    def latest(self) -> np.ndarray | None:
        """The most recent temperature field, whether or not it was kept as a frame."""
        with self._lock:
            return self._latest

    def frame(self, i: int) -> tuple[float, np.ndarray]:
        """Time and field of a stored frame, with the index clamped to the valid range."""
        with self._lock:
            i = max(0, min(i, len(self._frames) - 1))
            return self._frame_t[i], self._frames[i]

    def frame_times(self) -> np.ndarray:
        """Times of all stored frames."""
        with self._lock:
            return np.asarray(self._frame_t, float)

    def probe(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        """(times, temperature) of one cell, at frame resolution."""
        with self._lock:
            if not self._frames:
                return np.array([]), np.array([])
            return np.asarray(self._frame_t, float), np.array([f[node] for f in self._frames], float)

    def to_csv(self, path: str, probes: dict[str, int] | None = None) -> None:
        """Write the loop series to CSV, plus a second `_probes.csv` with the probed cells at frame
        resolution.
        """
        s = self.scalars()
        cols = ["t_d", "T_in_C", "T_out_C", "heat_removed_W", "Q_load_W", "pump_power_W"]
        data = np.column_stack([s[c] for c in cols]) if len(s["t_d"]) else np.zeros((0, len(cols)))
        np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
        if probes:
            t = self.frame_times()
            series = [self.probe(n)[1] for n in probes.values()]
            np.savetxt(path.replace(".csv", "_probes.csv"), np.column_stack([t] + series), delimiter=",",
                       header=",".join(["t_d"] + list(probes)), comments="", fmt="%.6g")


class BatchRecorder:
    """Stands in for History inside the simulation process: keeps no frames, just
    collects what changed since the last drain() into a StepBatch."""

    def __init__(self, max_frames: int = 600, session_id: int = 0):
        self.max_frames, self.session_id = max_frames, session_id
        self._schedule = FrameSchedule(max_frames)
        self.clear()

    def clear(self, initial_field: np.ndarray | None = None) -> None:
        """Start a new run, optionally with the starting field."""
        self._schedule.reset(has_initial=initial_field is not None)
        self._pending = {k: [] for k in SCALARS}
        self._frames: list[tuple[float, np.ndarray]] = []
        self._latest = None if initial_field is None else np.asarray(initial_field, np.float32)

    @property
    def n_steps(self) -> int:
        """Number of steps recorded since the run started."""
        return self._schedule.n

    @property
    def pending(self) -> int:
        """Steps collected since the last drain."""
        return len(self._pending["t_d"])

    def append(self, rec: StepRecord) -> None:
        """Collect one step, keeping its field only if the window's History will keep it as a frame."""
        for k in SCALARS:
            self._pending[k].append(getattr(rec, k))
        field = np.asarray(rec.field, np.float32)
        self._latest = field
        keep, _ = self._schedule.next_step()
        if keep:
            self._frames.append((rec.t_d, field))

    def latest(self) -> np.ndarray | None:
        """The most recent temperature field."""
        return self._latest

    def drain(self, with_latest: bool = True) -> StepBatch:
        """Package everything collected since the last drain as a StepBatch and start collecting
        afresh.
        """
        batch = StepBatch(self.session_id, {k: np.asarray(v, float) for k, v in self._pending.items()},
                          self._frames, self._latest if with_latest else None)
        self._pending = {k: [] for k in SCALARS}
        self._frames = []
        return batch

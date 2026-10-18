"""Qt-free live-run pieces: config, history, session, engine registry."""
import threading

import numpy as np
import pytest

from conftest import TINY_MINE
from mine_sandbox.engines import ENGINES
from mine_sandbox.live import History, LiveConfig, SimulationSession, build_scenario
from mine_sandbox.simulation import StepRecord


def _rec(t, n=10):
    """A fake StepRecord at time t with an n-cell field."""
    return StepRecord(t_d=t, T_in_C=35.0, T_out_C=30.0, heat_removed_W=1.0, Q_load_W=np.nan,
                      pump_power_W=np.nan, field=np.full(n, t))


def test_history_keeps_every_scalar_but_bounds_frames():
    h = History(max_frames=16)
    h.clear(initial_field=np.zeros(10))
    for i in range(1, 1001):
        h.append(_rec(float(i)))
    assert h.n_steps == 1000 and len(h.scalars()["t_d"]) == 1000
    assert h.n_frames <= 16
    t, field = h.frame(h.n_frames - 1)
    assert np.all(field == t)                        # frames stay matched to their times
    times, T = h.probe(3)
    assert np.allclose(times, T) and np.all(np.diff(times) > 0)


def test_history_is_safe_to_read_while_writing():
    h = History(max_frames=50)
    h.clear(initial_field=np.zeros(10))
    errors = []
    def writer():
        for i in range(1, 3000):
            h.append(_rec(float(i)))
    def reader():
        try:
            for _ in range(3000):
                s = h.scalars()
                assert len(s["t_d"]) == len(s["T_out_C"])
                h.probe(1)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    th = [threading.Thread(target=writer), threading.Thread(target=reader)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errors


def test_config_builds_drift_and_mine_scenarios():
    d = build_scenario(LiveConfig(source="drift", drift_grid="slab2d", length_m=200))
    assert "2-D slab" in d.grid.label
    m = build_scenario(LiveConfig(source="mine", mine=TINY_MINE, T_rock_C=None, flow_m3_h=100))
    assert m.operation.flow_m3_h == 100 and "tiny" in m.summary()
    with pytest.raises(ValueError):
        build_scenario(LiveConfig(source="mine", mine=None))


def test_session_steps_and_stops_at_the_end():
    s = SimulationSession(LiveConfig(source="mine", mine=TINY_MINE, sim_days=5, dt_days=0.5))
    assert s.step(3) == 3 and s.time_d == pytest.approx(1.5)
    assert s.step(100) == 7 and s.done and not s.runaway
    assert s.history.n_steps == 10


def test_session_reports_runaway():
    s = SimulationSession(LiveConfig(source="mine", mine=TINY_MINE, surface="fixed_load", load_MW=5.0,
                                     T_max_C=60.0, sim_days=200))
    s.step(10_000)
    assert s.runaway and s.time_d < 200


def test_engine_registry():
    assert ENGINES["fv"].available
    with pytest.raises(ValueError):
        SimulationSession(LiveConfig(engine="nope"))


# ---------------------------------------------------------------- batches (process <-> window)
from mine_sandbox.live import BatchRecorder, BuildProgress, FrameSchedule, SimulationProcess, StepBatch  # noqa: E402


def test_batches_rebuild_exactly_the_same_history():
    """The simulation process ships only the frames the window's History keeps; replaying
    the batches must give the same History as appending every step directly."""
    rng = np.random.default_rng(0)
    direct, remote = History(max_frames=16), History(max_frames=16)
    rec = BatchRecorder(max_frames=16, session_id=7)
    init = np.zeros(10)
    direct.clear(initial_field=init)
    remote.clear(initial_field=init)
    rec.clear(initial_field=init)
    i = 0
    while i < 1500:
        for _ in range(int(rng.integers(1, 60))):       # uneven batch sizes, as from a paced worker
            i += 1
            direct.append(_rec(float(i)))
            rec.append(_rec(float(i)))
        b = rec.drain()
        assert b.session_id == 7 and len(b.frames) <= b.n_steps
        remote.extend(b)
    assert remote.n_steps == direct.n_steps == i
    assert np.array_equal(remote.frame_times(), direct.frame_times())
    assert np.array_equal(remote.latest(), direct.latest())
    for k, v in direct.scalars().items():
        assert np.array_equal(remote.scalars()[k], v, equal_nan=True)
    assert np.array_equal(remote.probe(4)[1], direct.probe(4)[1])


def test_frame_schedule_matches_history_bounds():
    s = FrameSchedule(max_frames=8)
    kept = [s.next_step() for _ in range(100)]
    assert s.count <= 8 and s.stride > 1 and any(t for _, t in kept)


def test_a_batch_with_missing_frames_is_rejected():
    h = History(max_frames=8)
    h.clear(initial_field=np.zeros(3))
    bad = StepBatch(0, {k: np.zeros(2) for k in ("t_d", "T_in_C", "T_out_C", "heat_removed_W",
                                                  "Q_load_W", "pump_power_W")}, frames=[])
    with pytest.raises(StopIteration):
        h.extend(bad)


# ---------------------------------------------------------------- staged build + progress
def test_session_reports_build_stages_in_order():
    seen = []
    s = SimulationSession(LiveConfig(source="mine", mine=TINY_MINE, sim_days=5, device="cpu"),
                          progress=seen.append)
    keys = [p.stage for p in seen]
    assert keys == ["settings", "grid", "properties", "flow", "matrix", "solver", "surface", "done"]
    fr = [p.fraction() for p in seen]
    assert fr[-1] == 1.0 and all(0 <= f <= 1 for f in fr)
    assert all(b >= a - 1e-9 for a, b in zip(fr, fr[1:]))                    # never goes backwards
    assert seen[-1].finished and seen[-1].eta_s() == 0.0
    assert set(s.info.stage_seconds) == set(keys[:-1])
    assert s.info.build_seconds >= sum(s.info.stage_seconds.values()) * 0.99
    assert "CPU" in s.info.compute_label and s.info.speed_factor > 0


def test_progress_interpolates_inside_a_stage_and_admits_overruns():
    p = BuildProgress(stage="solver", label="Preparing the solver", index=5, count=7, done_s=1.0,
                      stage_s=2.0, total_s=4.0, elapsed_s=1.0)
    assert p.fraction(0.0) == pytest.approx(0.25)
    assert p.fraction(1.0) == pytest.approx(0.5)
    assert p.fraction(10.0) < 1.0                                            # a stage is never "over" early
    assert p.eta_s(0.5) == pytest.approx(2.5)
    assert p.eta_s(10.0) is None                                             # longer than expected


def test_auto_device_without_a_gpu_runs_on_the_cpu_and_says_why(monkeypatch):
    monkeypatch.setenv("MINE_SANDBOX_DEVICE", "cpu")
    from mine_sandbox.compute import detect_devices
    detect_devices(refresh=True)
    try:
        s = SimulationSession(LiveConfig(source="mine", mine=TINY_MINE, sim_days=2, device="auto"))
        assert s.info.device_label.startswith("CPU") and s.info.notices
    finally:
        monkeypatch.delenv("MINE_SANDBOX_DEVICE")
        detect_devices(refresh=True)


# ---------------------------------------------------------------- the simulation process
def _wait(proc, kinds, timeout=60.0):
    """Events up to and including the first one of `kinds` (later ones stay queued for the next call)."""
    import time
    pending = proc.__dict__.setdefault("_test_pending", [])
    t0, out = time.time(), []
    while time.time() - t0 < timeout:
        if not pending:
            pending.extend(proc.poll(timeout=0.05))
        while pending:
            ev = pending.pop(0)
            out.append(ev)
            if ev[0] in kinds:
                return ev, out
    raise AssertionError(f"no {kinds} event within {timeout} s (got {[e[0] for e in out]})")


def test_simulation_process_builds_plays_and_streams_history():
    proc = SimulationProcess()
    try:
        _wait(proc, {"hello"})
        dev, _ = _wait(proc, {"devices"})
        assert dev[1].devices[0].key == "cpu"
        sid = proc.build(LiveConfig(source="mine", mine=TINY_MINE, sim_days=20, dt_days=0.5))
        built, evs = _wait(proc, {"built", "failed"})
        assert built[0] == "built" and built[1] == sid, built
        assert [e[2].stage for e in evs if e[0] == "progress"][-1] == "done"
        h = History(600)
        h.clear(initial_field=built[3])
        proc.set_speed(float("inf"))
        proc.play()
        tick = None
        import time
        t0 = time.time()
        while time.time() - t0 < 60 and not (tick and tick.done):
            for ev in proc.poll(timeout=0.05):
                if ev[0] == "batch":
                    h.extend(ev[2])
                    tick = ev[3]
                elif ev[0] == "run_failed":
                    raise AssertionError(ev)
        assert tick.done and not tick.playing and tick.t_d == pytest.approx(20.0)
        assert h.n_steps == tick.n_steps == 40 and h.n_frames == 41
        assert np.isfinite(h.latest()).all() and h.latest().max() > 20                  # the workings warmed up
    finally:
        proc.close()
    assert not proc.alive


def test_simulation_process_reports_build_errors_and_can_be_killed():
    proc = SimulationProcess(detect_devices=False)
    try:
        sid = proc.build(LiveConfig(source="mine", mine=None))
        ev, _ = _wait(proc, {"failed", "built"})
        assert ev[0] == "failed" and ev[1] == sid and "open or generate a mine" in ev[2]
    finally:
        proc.kill()
    assert not proc.alive

"""Compute backends: the block-preconditioned solver, device detection and GPU fallback.

The GPU path runs the same code as the CPU path through an ArrayBackend, so
these tests exercise its logic on numpy. A fake "GPU" backend (numpy arrays,
device kind "cuda") checks that failures fall back to the CPU and say why.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import splu

from conftest import TINY_MINE
from mine_sandbox import ConstantInlet, CoupledStepper, FiniteVolumeModel, StructuredDriftGrid, assemble, make_scenario
from mine_sandbox import fv
from mine_sandbox.compute import (CPU, ArrayBackend, BlockPreconditionedSolver, ComputeDevice, DeviceReport,
                                  SolverError, detect_devices, resolve_device)
from mine_sandbox.mines import scenario_from_mine

FAKE_GPU = ComputeDevice("cuda:0", "GPU 0: Fake (numpy)", kind="cuda", index=0, memory_gb=1.0)


def _problem(kind="mine", days=5):
    """A small assembled problem on the tiny mine or one of the drift grids."""
    if kind == "mine":
        return assemble(scenario_from_mine(TINY_MINE, sim_days=days))
    return assemble(make_scenario(width_m=6, height_m=2, diameter_m=None, sim_days=days,
                                  grid=StructuredDriftGrid.block_3d(n_x=8, n_side=6, n_core=5)))


def _run(model, T_in=35.0):
    """Step a model to the end at a fixed inlet temperature and return the final field."""
    st = CoupledStepper(model, ConstantInlet(T_in))
    while st.step() is not None:
        pass
    return model.temperatures()


@pytest.mark.parametrize("kind", ["mine", "block3d"])
def test_block_solver_matches_lu_on_one_solve(kind):
    m = FiniteVolumeModel(_problem(kind), solver="direct")
    b = m._rhs()
    lu = splu(m._M.tocsc())
    exact = lu.solve(b)
    for _ in range(2):              # refine: plain LU is ~1e-6 off on the far-field rows (scaling)
        exact = exact + lu.solve(b - m._M @ exact)
    s = BlockPreconditionedSolver(m._M, m.problem.graph.is_drift, rtol=1e-13)
    x = s.solve(b, np.full_like(b, 13.0))
    assert np.max(np.abs(x - exact)) < 1e-8
    assert s.last_iterations <= 30


@pytest.mark.parametrize("kind", ["mine", "block3d"])
def test_block_and_direct_solvers_agree_over_a_run(kind):
    p = _problem(kind)
    a = _run(FiniteVolumeModel(p, solver="direct"))
    b = _run(FiniteVolumeModel(p, solver="block"))
    assert np.max(np.abs(a - b)) < 1e-5


def test_block_solver_needs_few_iterations_on_a_mine():
    m = FiniteVolumeModel(_problem("mine"), solver="block")
    for _ in range(6):
        m.advance()
    assert m.last_iterations is not None and m.last_iterations <= 6


def test_auto_solver_picks_by_size():
    assert isinstance(FiniteVolumeModel(_problem("mine")).__dict__["_solver"], fv.DirectSolver)     # 2.4k cells
    big = FiniteVolumeModel(assemble(make_scenario(width_m=6, height_m=2, diameter_m=None, sim_days=1,
                                                   grid=StructuredDriftGrid.block_3d(n_x=12, n_side=8, n_core=5))))
    assert big.problem.graph.n_cells > fv.DIRECT_LIMIT
    assert isinstance(big._solver, fv.BlockSolver) and "block" in big.solver_description


def test_cpu_report_and_resolution_without_a_gpu(monkeypatch):
    report = DeviceReport(devices=(CPU,), gpu_reason="CuPy is not installed")
    assert resolve_device("cpu", report) == (CPU, None)
    dev, why = resolve_device("auto", report)
    assert dev == CPU and "CuPy" in why
    dev, why = resolve_device("cuda:1", report)
    assert dev == CPU and why
    with_gpu = DeviceReport(devices=(CPU, FAKE_GPU))
    assert resolve_device("auto", with_gpu) == (FAKE_GPU, None)
    assert resolve_device("gpu", with_gpu) == (FAKE_GPU, None)


def test_environment_variable_forces_the_cpu(monkeypatch):
    monkeypatch.setenv("MINE_SANDBOX_DEVICE", "cpu")
    rep = detect_devices(refresh=True)
    assert rep.devices == (CPU,) and "MINE_SANDBOX_DEVICE" in rep.gpu_reason
    monkeypatch.delenv("MINE_SANDBOX_DEVICE")
    detect_devices(refresh=True)


class _FakeGpuBackend(ArrayBackend):
    """numpy arrays pretending to be a GPU; can be told to fail after a number of norms."""
    def __init__(self, fail_after=None):
        super().__init__(FAKE_GPU)
        self.calls, self.fail_after = 0, fail_after

    def norm(self, a):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("cudaErrorIllegalAddress: an illegal memory access was encountered")
        return super().norm(a)


def _fake_gpu(monkeypatch, backend_factory):
    """Make the fv engine believe a GPU exists, using the given backend in its place."""
    monkeypatch.setattr(fv, "resolve_device", lambda key: (FAKE_GPU, None))
    monkeypatch.setattr(fv, "make_backend", lambda device: backend_factory())


def test_gpu_path_runs_and_matches_the_cpu(monkeypatch):
    p = _problem("block3d")
    ref = _run(FiniteVolumeModel(p, solver="direct"))
    _fake_gpu(monkeypatch, _FakeGpuBackend)
    m = FiniteVolumeModel(p, device="cuda:0")
    assert m.device.is_gpu and m.fallback_reason is None and "GPU" in m.solver_description
    assert np.max(np.abs(_run(m) - ref)) < 1e-5
    assert not m.notices


def test_gpu_that_fails_to_build_falls_back_to_the_cpu(monkeypatch):
    def broken():
        raise RuntimeError("out of memory allocating 512 MB")
    _fake_gpu(monkeypatch, broken)
    m = FiniteVolumeModel(_problem("block3d"), device="cuda:0")
    assert not m.device.is_gpu and "out of memory" in m.fallback_reason
    assert "CPU" in m.solver_description


def test_gpu_that_fails_mid_run_carries_on_on_the_cpu(monkeypatch):
    p = _problem("block3d", days=10)
    ref = _run(FiniteVolumeModel(p, solver="direct"))
    _fake_gpu(monkeypatch, _FakeGpuBackend)
    m = FiniteVolumeModel(p, device="cuda:0")
    assert m.device.is_gpu
    m._xb.fail_after = m._xb.calls + 40                                 # a few steps in, the "GPU" dies
    out = _run(m)
    assert not m.device.is_gpu and len(m.notices) == 1 and "illegal memory access" in m.notices[0]
    assert np.max(np.abs(out - ref)) < 1e-5                              # no step lost or repeated


def test_explicit_cpu_solvers_stay_on_the_cpu(monkeypatch):
    _fake_gpu(monkeypatch, _FakeGpuBackend)
    m = FiniteVolumeModel(_problem("mine"), device="auto", solver="direct")
    assert not m.device.is_gpu and "CPU only" in m.fallback_reason


def test_solver_errors_are_reported():
    m = FiniteVolumeModel(_problem("block3d"), solver="direct")
    s = BlockPreconditionedSolver(m._M, m.problem.graph.is_drift, maxiter=1, rtol=1e-30)
    with pytest.raises(SolverError):
        s.solve(m._rhs(), np.zeros(m.problem.graph.n_cells))


def test_auto_keeps_the_faster_device(monkeypatch):
    """device="auto" with a GPU times both on the first step and keeps the faster one."""
    class SlowGpu(_FakeGpuBackend):
        def norm(self, a):
            import time
            time.sleep(0.002)                     # a "GPU" that is slower than the CPU here
            return super().norm(a)
    _fake_gpu(monkeypatch, SlowGpu)
    m = FiniteVolumeModel(_problem("block3d"), device="auto")
    assert not m.device.is_gpu and m.selection_note.startswith("Auto chose the CPU")
    _fake_gpu(monkeypatch, _FakeGpuBackend)
    m = FiniteVolumeModel(_problem("block3d"), device="cuda:0")         # explicit: no comparison
    assert m.device.is_gpu and m.selection_note is None


def test_cupy_backend_wiring_with_a_numpy_stand_in(monkeypatch):
    """Runs CupyBackend's own code (transfers, CSR construction, device selection) with numpy
    standing in for CuPy, so a typo there can't hide until someone has a GPU."""
    import sys
    import types

    import scipy.sparse as ssp
    fake = types.ModuleType("cupy")
    fake.float64, fake.int32, fake.int64 = np.float64, np.int32, np.int64
    fake.asarray = lambda a, dtype=None: np.array(a, dtype=dtype)
    fake.asnumpy = lambda a: np.array(a)
    fake.linalg = np.linalg
    fake.dot, fake.zeros_like = np.dot, np.zeros_like
    used = []
    class _Device:
        def __init__(self, i):
            self.i = i
        def use(self):
            used.append(self.i)
    class _Stream:
        @staticmethod
        def synchronize():
            pass
    fake.cuda = types.SimpleNamespace(Device=_Device, Stream=types.SimpleNamespace(null=_Stream))
    cupyx = types.ModuleType("cupyx")
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_scipy.sparse = ssp
    cupyx.scipy = cupyx_scipy
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.sparse", ssp)
    from mine_sandbox.compute.devices import CupyBackend
    p = _problem("block3d")
    ref = _run(FiniteVolumeModel(p, solver="direct"))
    monkeypatch.setattr(fv, "resolve_device", lambda key: (FAKE_GPU, None))
    monkeypatch.setattr(fv, "make_backend", lambda device: CupyBackend(device))
    m = FiniteVolumeModel(p, device="cuda:0")
    assert used == [0] and m.device.is_gpu and isinstance(m._xb, CupyBackend)
    assert np.max(np.abs(_run(m) - ref)) < 1e-5

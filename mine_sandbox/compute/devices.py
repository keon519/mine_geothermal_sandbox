"""Where the heat solve runs: the CPU (numpy/scipy) or an NVIDIA GPU (CuPy).

The same idea as a FAISS index that can live on the GPU or the CPU: one small
array interface (ArrayBackend) with two implementations, device detection
that actually *runs* a kernel before offering a GPU, and a CPU fallback that
says why whenever the GPU can't be used.

    detect_devices()        -> DeviceReport (cached; probes each CUDA device)
    resolve_device("auto")  -> (ComputeDevice, reason the GPU wasn't used or None)
    make_backend(device)    -> ArrayBackend

Device keys: "auto" (GPU if one works, else CPU), "cpu", "cuda:0", "cuda:1", ...
Set MINE_SANDBOX_DEVICE=cpu to force the CPU everywhere (handy when debugging).

GPU support needs CuPy built for your CUDA driver, e.g.
    pip install "cupy-cuda12x[ctk]"      (or cupy-cuda13x[ctk] with a CUDA 13 driver)
Nothing else in the package imports CuPy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

CPU_KEY = "cpu"
AUTO_KEY = "auto"
CUPY_HINT = 'pip install "cupy-cuda12x[ctk]"  (or cupy-cuda13x[ctk] for a CUDA 13 driver)'


@dataclass(frozen=True)
class ComputeDevice:
    """A place the heat solve can run: the CPU, or one specific NVIDIA GPU."""
    key: str                  # "cpu" or "cuda:<n>"
    label: str                # what the settings panel shows
    kind: str = "cpu"         # "cpu" | "cuda"
    index: int = 0
    memory_gb: float = 0.0

    @property
    def is_gpu(self) -> bool:
        """True for a CUDA GPU."""
        return self.kind == "cuda"


CPU = ComputeDevice(CPU_KEY, f"CPU ({os.cpu_count() or 1} threads)")


@dataclass(frozen=True)
class DeviceReport:
    """What compute hardware was found, and why a GPU isn't on offer if none is.

    The CPU is always listed first. The report also keeps per-device problems that didn't stop
    the others from working, so the GUI can explain exactly what happened.
    """
    devices: tuple[ComputeDevice, ...] = (CPU,)
    gpu_reason: str | None = None          # why no GPU is offered (None when at least one works)
    probe_seconds: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)   # per-device problems that didn't stop the rest

    @property
    def gpus(self) -> tuple[ComputeDevice, ...]:
        """Just the working GPUs."""
        return tuple(d for d in self.devices if d.is_gpu)

    def get(self, key: str) -> ComputeDevice | None:
        """Look up a device by its key, or None if it wasn't found."""
        return next((d for d in self.devices if d.key == key), None)


_REPORT: DeviceReport | None = None


def _probe_cuda() -> tuple[list[ComputeDevice], str | None, list[str]]:
    """Find CUDA devices that can really run this package's kernels.

    It isn't enough for CuPy to import or for a GPU to show up: drivers and CuPy builds can be
    mismatched in ways that only fail when a kernel actually runs. So each GPU gets a tiny test
    solve using the same kinds of operations as the real solver, and only GPUs that return the
    right answer are offered.

    Returns (working devices, reason none work or None, notes on devices that failed).
    """
    try:
        import cupy
        import cupyx.scipy.sparse as csp
    except ImportError:
        return [], f"CuPy is not installed — {CUPY_HINT}", []
    except Exception as e:  # noqa: BLE001 - a broken CUDA install raises all sorts of things
        return [], f"CuPy failed to load ({type(e).__name__}: {e})", []
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as e:  # noqa: BLE001
        return [], f"no usable NVIDIA driver/GPU ({type(e).__name__}: {e})", []
    if count == 0:
        return [], "no CUDA device found", []
    found, notes = [], []
    for i in range(count):
        name = f"GPU {i}"
        try:
            props = cupy.cuda.runtime.getDeviceProperties(i)
            raw = props.get("name", b"")
            name = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
            mem = props.get("totalGlobalMem", 0) / 2**30
            cc = f"{props.get('major', '?')}.{props.get('minor', '?')}"
            with cupy.cuda.Device(i):
                # compile and run the kinds of kernels the solver uses (elementwise, reduction, SpMV)
                x = cupy.arange(1, 9, dtype=cupy.float64)
                A = csp.csr_matrix((cupy.full(8, 2.0), cupy.arange(8, dtype=cupy.int32),
                                    cupy.arange(9, dtype=cupy.int32)), shape=(8, 8))
                y = A @ x
                ok = abs(float(cupy.dot(y, x)) - 2.0 * 204.0) < 1e-9
                cupy.cuda.Stream.null.synchronize()
            if not ok:
                raise RuntimeError("test kernel gave a wrong answer")
            found.append(ComputeDevice(f"cuda:{i}", f"GPU {i}: {name} ({mem:.0f} GB, CC {cc})",
                                       kind="cuda", index=i, memory_gb=mem))
        except Exception as e:  # noqa: BLE001
            notes.append(f"{name}: {type(e).__name__}: {e}")
    if not found:
        why = "; ".join(notes) or "unknown error"
        return [], f"CUDA device(s) present but not usable with this CuPy build ({why})", notes
    return found, None, notes


def detect_devices(refresh: bool = False) -> DeviceReport:
    """CPU plus every CUDA device that passes a test kernel. Cached after the first call."""
    global _REPORT
    if _REPORT is not None and not refresh:
        return _REPORT
    import time
    t0 = time.perf_counter()
    if os.environ.get("MINE_SANDBOX_DEVICE", "").strip().lower() == CPU_KEY:
        gpus, reason, notes = [], "GPU disabled by MINE_SANDBOX_DEVICE=cpu", []
    else:
        gpus, reason, notes = _probe_cuda()
    _REPORT = DeviceReport(devices=(CPU, *gpus), gpu_reason=reason,
                           probe_seconds=time.perf_counter() - t0, notes=tuple(notes))
    return _REPORT


def resolve_device(key: str | None, report: DeviceReport | None = None) -> tuple[ComputeDevice, str | None]:
    """Turn a device key into a usable device. Returns (device, why the GPU wasn't used).

    "auto" takes the first working GPU, else the CPU (with the reason). An explicit
    "cuda:n" that doesn't work also falls back to the CPU, with the reason."""
    key = (key or AUTO_KEY).strip().lower()
    if key == CPU_KEY:
        return CPU, None
    if key == "cuda" or key == "gpu":
        key = "cuda:0"
    rep = report or detect_devices()
    if key == AUTO_KEY:
        return (rep.gpus[0], None) if rep.gpus else (CPU, rep.gpu_reason)
    dev = rep.get(key)
    if dev is not None:
        return dev, None
    why = rep.gpu_reason or f"{key} is not available (found: {', '.join(d.key for d in rep.devices)})"
    return CPU, why



# Array backends: the handful of operations the solver needs

class ArrayBackend:
    """The small set of array operations the solver needs, implemented with numpy/scipy on the CPU.

    The solver is written once against this interface. On the CPU the transfers are no-ops; the
    GPU backend overrides them (and the sparse matrix type) to move data to and from the card,
    so the same solver code runs in both places.
    """
    name = "numpy"

    def __init__(self, device: ComputeDevice = CPU):
        self.device = device
        self.xp = np

    # transfers (no-ops on the CPU)
    def asarray(self, a) -> np.ndarray:
        """Bring a float array onto this backend."""
        return np.asarray(a, dtype=np.float64)

    def asindex(self, a) -> np.ndarray:
        """Bring an integer index array onto this backend."""
        return np.asarray(a, dtype=np.int64)

    def to_host(self, a) -> np.ndarray:
        """Copy an array back to the host as numpy."""
        return np.asarray(a)

    def csr(self, M):
        """Convert a scipy sparse matrix to this backend's CSR type."""
        return M.tocsr()

    # reductions that return Python floats (on the GPU these synchronise)
    def norm(self, a) -> float:
        """Euclidean norm as a Python float (forces a GPU sync)."""
        return float(np.linalg.norm(a))

    def item(self, a, i: int) -> float:
        """One element as a Python float, without copying the whole array to the host."""
        return float(a[i])

    def synchronize(self) -> None:
        """Wait for queued work to finish, so timings are honest. Nothing to wait for on the CPU."""
        pass


class CupyBackend(ArrayBackend):
    """The same operations on an NVIDIA GPU through CuPy."""
    name = "cupy"

    def __init__(self, device: ComputeDevice):
        import cupy
        import cupyx.scipy.sparse as csp
        super().__init__(device)
        self._cupy, self._csp = cupy, csp
        cupy.cuda.Device(device.index).use()          # the solver runs on one thread: make it current
        self.xp = cupy

    def asarray(self, a):
        return self._cupy.asarray(np.asarray(a, dtype=np.float64))

    def asindex(self, a):
        return self._cupy.asarray(np.asarray(a, dtype=np.int64))

    def to_host(self, a) -> np.ndarray:
        return self._cupy.asnumpy(a)

    def csr(self, M):
        M = M.tocsr()
        M.sort_indices()
        cp = self._cupy
        return self._csp.csr_matrix((cp.asarray(M.data, dtype=cp.float64), cp.asarray(M.indices, dtype=cp.int32),
                                     cp.asarray(M.indptr, dtype=cp.int32)), shape=M.shape)

    def norm(self, a) -> float:
        return float(self._cupy.linalg.norm(a))

    def item(self, a, i: int) -> float:
        return float(a[i])

    def synchronize(self) -> None:
        self._cupy.cuda.Stream.null.synchronize()


def make_backend(device: ComputeDevice) -> ArrayBackend:
    """The right array backend for a device: CuPy for a GPU, numpy otherwise."""
    return CupyBackend(device) if device.is_gpu else ArrayBackend(device)

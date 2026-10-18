"""Compute backends: run the heat solve on the CPU or an NVIDIA GPU (CuPy), with fallback."""
from .block_solver import BlockPreconditionedSolver, SolverError
from .devices import (AUTO_KEY, CPU, CPU_KEY, CUPY_HINT, ArrayBackend, ComputeDevice, DeviceReport,
                      detect_devices, make_backend, resolve_device)

__all__ = ["BlockPreconditionedSolver", "SolverError", "AUTO_KEY", "CPU", "CPU_KEY", "CUPY_HINT",
           "ArrayBackend", "ComputeDevice", "DeviceReport", "detect_devices", "make_backend", "resolve_device"]

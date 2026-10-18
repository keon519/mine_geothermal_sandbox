"""Registry of ThermalModel engines, so UIs and scripts can offer a choice
without importing (or depending on) any particular solver.

Register a new engine with register_engine(); it appears in the GUI's engine
list automatically.

A factory is called as factory(problem, **options). Options an engine may use:
    device="auto"|"cpu"|"cuda:0"    where to solve (engines with supports_device)
    progress=callable(stage_key)    reports the stages listed in `stages`
Factories ignore options they don't understand, so `model_factory=info.factory`
still works in run_coupled().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .assembly import ThermalProblem
from .model import ThermalModel


@dataclass(frozen=True)
class EngineInfo:
    """Everything a UI needs to know to offer a solver as a choice.

    Besides the factory that builds the model, it carries a check that explains why an engine
    can't be used on this machine (for example, MODFLOW isn't installed), the list of build
    stages it will report so progress bars know what to expect, and whether it can run on a GPU.
    """
    key: str
    label: str
    factory: Callable[..., ThermalModel]
    check: Callable[[], str | None]          # None if usable, else the reason it isn't
    note: str = ""
    stages: tuple[str, ...] = ("engine",)    # progress stage keys the factory reports, in order
    supports_device: bool = False            # can run on a GPU (device= option)

    @property
    def available(self) -> bool:
        """True if the engine can be used on this machine right now."""
        return self.check() is None


ENGINES: dict[str, EngineInfo] = {}


def register_engine(info: EngineInfo) -> None:
    """Add an engine to the registry (or replace one with the same key) so it shows up wherever
    engines are listed.
    """
    ENGINES[info.key] = info


def _fv_factory(problem: ThermalProblem, device: str = "cpu", progress=None, solver: str = "auto", **_):
    """Build the built-in finite-volume model, which can run on the CPU or a GPU and reports its
    build stages.
    """
    from .fv import FiniteVolumeModel
    return FiniteVolumeModel(problem, solver=solver, device=device, progress=progress)


def _mf6_factory(problem: ThermalProblem, progress=None, **_):
    """Start a MODFLOW 6 session through its shared library. It does its setup in one go, so it
    reports a single stage.
    """
    from .mf6 import Mf6ApiSession
    if progress:
        progress("engine")
    return Mf6ApiSession(problem)


def _mf6_check():
    """Explain why MODFLOW can't be used here (missing Python package, executable or library), or
    return None if it can.
    """
    from importlib.util import find_spec
    if find_spec("modflowapi") is None:
        return "modflowapi is not installed (pip install modflowapi)"
    try:
        from .mf6 import find_libmf6, find_mf6
        find_mf6()
        find_libmf6()
    except FileNotFoundError as e:
        return str(e)
    return None


register_engine(EngineInfo("fv", "Built-in finite volume (fast)", _fv_factory, lambda: None,
                           "scipy solver, or CuPy on an NVIDIA GPU; matches MODFLOW to ~1e-4 °C",
                           stages=("flow", "matrix", "solver"), supports_device=True))
register_engine(EngineInfo("mf6", "MODFLOW 6 (libmf6)", _mf6_factory, _mf6_check,
                           "the reference solver; CPU only, slower on large mines"))

"""Mine-water heat-sink sandbox: MODFLOW 6 (the mine) + TESPy (the surface loop).

Layers, from the inside out - each depends only on the ones above it:

    materials, geometry, geology      what things are made of and shaped like
    grids                             how space is cut into cells  (Grid -> CellGraph)
    scenario                          what to simulate
    assembly, energy, results         solver-neutral physics and bookkeeping
    model, surface                    the two interfaces the coupler talks to
    mf6                               MODFLOW 6 implementations (the only flopy code)
    fv                                built-in finite-volume ThermalModel (scipy)
    simulation                        run_fixed_inlet / run_coupled
    plotting                          figures from a RunResult
"""
import sys as _sys

try:                                   # a broken CuPy/CUDA install must not take the app down
    import cupy as _cupy  # noqa: F401
except Exception:                      # noqa: BLE001 - e.g. a CUDA DLL that won't load
    _sys.modules["cupy"] = None        # TESPy (and we) then see "CuPy not installed" and use the CPU
from .assembly import CellProperties, GeologyNotSupportedError, ThermalProblem, assemble
from .geology import Geology, Layer, LayeredGeology, UniformGeology
from .geometry import CircularSection, CrossSection, DriftSpec, RectangularSection
from .grids import CellGraph, Grid, RadialDriftGrid, StructuredDriftGrid
from .materials import ROCKS, WATER, Rock, get_rock
from .model import ThermalModel
from .results import RunResult
from .scenario import GroundSpec, InletPeriod, NumericsSpec, OperatingSpec, Scenario, make_scenario
from .fv import FiniteVolumeModel
from .simulation import CoupledStepper, StepRecord, run_coupled, run_fixed_inlet
from .surface import (ConstantInlet, FixedHeatLoad, FixedInletTemperature, InletController,
                      LoopState, TespyLoopSpec, TespySurfaceLoop)

__all__ = [
    "CellProperties", "GeologyNotSupportedError", "ThermalProblem", "assemble",
    "Geology", "Layer", "LayeredGeology", "UniformGeology",
    "CircularSection", "CrossSection", "DriftSpec", "RectangularSection",
    "CellGraph", "Grid", "RadialDriftGrid", "StructuredDriftGrid",
    "ROCKS", "WATER", "Rock", "get_rock", "ThermalModel", "RunResult",
    "GroundSpec", "InletPeriod", "NumericsSpec", "OperatingSpec", "Scenario", "make_scenario",
    "run_coupled", "run_fixed_inlet", "CoupledStepper", "StepRecord", "FiniteVolumeModel",
    "ConstantInlet", "FixedHeatLoad", "FixedInletTemperature", "InletController", "LoopState",
    "TespyLoopSpec", "TespySurfaceLoop",
]

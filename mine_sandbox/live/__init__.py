"""Live (step-by-step) runs, independent of any GUI toolkit.

    config     LiveConfig: what the settings panel picked
    session    SimulationSession: build (staged, with progress) then step on demand
    progress   BuildPipeline / BuildProgress: build stages, timings, time left
    history    History (what the window keeps), FrameSchedule, StepBatch, BatchRecorder
    process    SimulationProcess: a session in its own process, driven by commands
"""
from .config import DRIFT_GRIDS, SURFACE_MODES, LiveConfig, build_controller, build_scenario
from .history import BatchRecorder, FrameSchedule, History, StepBatch
from .process import MAX_SPEED, SimulationProcess, TickInfo
from .progress import BuildPipeline, BuildProgress
from .session import BuildInfo, SimulationSession

__all__ = ["DRIFT_GRIDS", "SURFACE_MODES", "LiveConfig", "build_controller", "build_scenario",
           "BatchRecorder", "FrameSchedule", "History", "StepBatch", "MAX_SPEED", "SimulationProcess",
           "TickInfo", "BuildPipeline", "BuildProgress", "BuildInfo", "SimulationSession"]

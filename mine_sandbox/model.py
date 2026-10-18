"""The abstraction the coupling code depends on (dependency inversion).

A ThermalModel is anything that can be advanced in time with a given inlet
temperature and report its temperatures back. The MODFLOW API session and
the file-based restart session both implement it; so could a pure-numpy
solver, or a future MODFLOW model with groundwater flow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .assembly import ThermalProblem


class ThermalModel(ABC):
    """A heat model of the mine that the surface loop can drive one step at a time.

    The coupling works like a conversation: the surface side says what temperature the water is
    going in at, the model advances one time step, and then reports what temperature comes back
    out at Pump Out. Anything that can hold up its end of that exchange can be plugged into the
    coupler, whether it's MODFLOW, the built-in finite-volume solver, or something new. Models
    are also context managers so that external processes or libraries get cleaned up.
    """
    problem: ThermalProblem

    @property
    @abstractmethod
    def time_d(self) -> float:
        """Simulated time reached so far, in days."""

    @property
    @abstractmethod
    def end_time_d(self) -> float:
        """Simulated time at which the run is scheduled to stop, in days."""

    @abstractmethod
    def set_inlet_temperature(self, T_C: float) -> None:
        """Temperature of the water entering at Pump In for the next advance()."""

    @abstractmethod
    def advance(self) -> None:
        """Move forward by the model's own step."""

    @abstractmethod
    def temperatures(self) -> np.ndarray:
        """Copy of the current temperature of every cell."""

    def outlet_temperature(self) -> float:
        """Temperature of the water reaching Pump Out, which is what goes back up to the surface."""
        return float(self.temperatures()[self.problem.graph.outlet_cell])

    @property
    def finished(self) -> bool:
        """True once the model has reached its end time (with a small tolerance for rounding)."""
        return self.time_d >= self.end_time_d - 1e-9

    def close(self) -> None:
        """Release anything the model holds open, such as a MODFLOW library session. Does nothing
        by default.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

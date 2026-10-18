"""The simplest surface: send water back at a fixed temperature."""
from __future__ import annotations

from dataclasses import dataclass

from .base import InletController, LoopState


@dataclass(frozen=True)
class ConstantInlet(InletController):
    """Always send water down at the same temperature, whatever comes back. Useful for fixed-inlet
    studies.
    """
    T_in_C: float

    def inlet_state(self, T_out_C: float, t_d: float) -> LoopState:
        return LoopState(T_in_C=self.T_in_C)

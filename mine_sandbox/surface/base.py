"""What sits above ground, seen from the coupling loop.

An InletController answers one question: given the water leaving the mine
(T_out) at time t, what goes back in? Anything that can answer it - a fixed
temperature, TESPy, a lookup table of reactor output, a controller that
switches between mine zones - plugs into run_coupled() unchanged.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopState:
    """What the surface side decided for one step: the inlet temperature, plus the heat load and
    pump power when they are modelled.
    """
    T_in_C: float                      # temperature sent back into the drift
    Q_load_W: float = math.nan         # heat added at the surface (NaN if not modelled)
    pump_power_W: float = math.nan     # pump electrical/shaft power (NaN if not modelled)


class InletController(ABC):
    """The surface half of the closed loop.

    Each step it is told the temperature of the water coming up from Pump Out and the current
    time, and it answers with the temperature that goes back down at Pump In.
    """
    @abstractmethod
    def inlet_state(self, T_out_C: float, t_d: float) -> LoopState:
        """Decide what goes back into the mine, given the returning water temperature and the time
        in days.
        """

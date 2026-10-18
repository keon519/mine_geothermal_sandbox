"""Energy bookkeeping, independent of any solver."""
from __future__ import annotations

import numpy as np

from .assembly import ThermalProblem
from .materials import WATER


def advected_heat_W(mass_flow_kg_s: float, T_in_C, T_out_C):
    """Heat the drift takes out of the water:  m_dot * c_p * (T_in - T_out)."""
    return mass_flow_kg_s * WATER.cp * (np.asarray(T_in_C) - np.asarray(T_out_C))


def stored_heat_J(problem: ThermalProblem, temperatures: np.ndarray) -> float:
    """Heat held in the water + rock relative to the undisturbed rock temperature:
    sum over cells of  V * (theta rho_w c_w + (1-theta) rho_s c_s) * (T - T_rock)."""
    dT = np.asarray(temperatures) - problem.scenario.ground.T_rock_C
    return float(np.sum(problem.graph.volumes * problem.props.volumetric_heat_capacity * dT))


def effectiveness(T_in_C, T_out_C, T_rock_C):
    """Fraction of the possible cooling achieved: (T_in - T_out) / (T_in - T_rock)."""
    T_in = np.asarray(T_in_C)
    return (T_in - np.asarray(T_out_C)) / (T_in - T_rock_C)

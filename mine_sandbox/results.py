"""One result type for every kind of run and every grid.

Temperatures are stored per cell (flat node arrays), and all spatial views
(along the drift, rock profile, vertical section) are asked of the CellGraph,
so plots and analysis work unchanged on 1-D, 2-D and 3-D grids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import energy
from .assembly import ThermalProblem


@dataclass
class RunResult:
    """The time history of a run: loop temperatures, heat removed, and the full temperature field.

    Every recorded step keeps the temperature of every cell, so any view can be produced
    afterwards: along the water route, out into the rock, or as a vertical section. Coupled runs
    also carry the surface heat load and pump power, and `runaway` says whether the run had to
    stop because the loop overheated.
    """
    problem: ThermalProblem
    times_d: np.ndarray            # end of each recorded step
    T_in_C: np.ndarray
    T_out_C: np.ndarray
    heat_removed_W: np.ndarray     # m_dot c_p (T_in - T_out)
    temperatures: np.ndarray       # (n_steps, n_cells)
    Q_load_W: np.ndarray | None = None       # surface heat load (coupled runs)
    pump_power_W: np.ndarray | None = None   # coupled runs with TESPy
    runaway: bool = False                    # stopped early: loop water too hot

    # basics
    @property
    def scenario(self):
        """The scenario this run was built from."""
        return self.problem.scenario

    @property
    def graph(self):
        """The cell graph the temperatures refer to."""
        return self.problem.graph

    @property
    def effectiveness(self) -> np.ndarray:
        """Cooling achieved at each step as a fraction of the most possible, (T_in - T_out) / (T_in
        - T_rock).
        """
        return energy.effectiveness(self.T_in_C, self.T_out_C, self.scenario.ground.T_rock_C)

    def at_day(self, day: float) -> int:
        """Index of the recorded step closest to a given day."""
        return int(np.argmin(np.abs(self.times_d - day)))

    def stored_heat_J(self, step: int = -1) -> float:
        """Heat the ground has absorbed by a given step, relative to undisturbed rock."""
        return energy.stored_heat_J(self.problem, self.temperatures[step])

    # spatial views
    def along_drift(self, step: int = -1):
        """(x, water temperature) from Pump In to Pump Out."""
        x, nodes = self.graph.along_drift()
        return x, self.temperatures[step, nodes]

    def wall_profile(self, step: int = -1, at_fraction: float = 0.5):
        """(distance from drift axis, rock temperature) moving away from the wall."""
        d, nodes = self.graph.wall_profile(at_fraction)
        return d, self.temperatures[step, nodes]

    def section(self, step: int = -1):
        """(x centres, cross-axis centres, T[cross, x]) in the vertical plane through
        the middle of the main route."""
        lat = self.graph.lattice
        if lat is not None:
            mid = int(self.graph.path_cells[len(self.graph.path_cells) // 2])
            _, j, _ = lat.voxel_of(mid)
            ids = lat.row(j)                                   # (nx, nz)
            img = np.where(ids >= 0, self.temperatures[step][np.maximum(ids, 0)], np.nan)
            return lat.centres(lat.x_edges), lat.centres(lat.z_edges), img.T
        nodes = self.graph.section_plane()
        c = self.graph.centroids[nodes]
        xs, xi = np.unique(np.round(c[:, 0], 6), return_inverse=True)
        zs, zi = np.unique(np.round(c[:, 2], 6), return_inverse=True)
        img = np.full((len(zs), len(xs)), np.nan)
        img[zi, xi] = self.temperatures[step, nodes]
        return xs, zs, img

    # summary
    def summary(self) -> str:
        """Readable summary of the scenario and the final state of the run, flagging a runaway if
        one happened.
        """
        head = self.scenario.summary()
        if self.runaway:
            head += (f"\n  ** RUNAWAY: loop water passed the limit at t={self.times_d[-1]:.0f} d "
                     "- this drift cannot reject this load **")
        head += f", residence={self.problem.residence_time_d * 24:.1f} h"
        line = (f"\n  t={self.times_d[-1]:.0f} d: T_in={self.T_in_C[-1]:.2f} C, "
                f"T_out={self.T_out_C[-1]:.2f} C, removed={self.heat_removed_W[-1] / 1e6:.3f} MW, "
                f"effectiveness={100 * self.effectiveness[-1]:.1f}%")
        if self.Q_load_W is not None and np.isfinite(self.Q_load_W[-1]):
            line += f", load={self.Q_load_W[-1] / 1e6:.3f} MW"
        if self.pump_power_W is not None and np.isfinite(self.pump_power_W[-1]):
            line += f", pump={self.pump_power_W[-1] / 1e3:.1f} kW"
        return head + line

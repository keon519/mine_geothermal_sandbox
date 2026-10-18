"""Axisymmetric grid: a line of water cells along the drift, each wrapped in
concentric rings of rock.


Cheap (hundreds of cells) and accurate for a single drift in uniform rock, but
it cannot tell up from sideways, so it rejects layered geology
(`resolves_depth = False`). Use StructuredDriftGrid for layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import DriftSpec
from .base import CellGraph, FaceList, Grid, Lattice, rectangle_layout


@dataclass(frozen=True)
class RadialDriftGrid(Grid):
    """A single drift as a row of water cells, each surrounded by rings of rock.

    Because heat spreads out from a tunnel roughly the same in every direction, the rock can be
    represented by concentric rings rather than a full 3-D block. The rings grow geometrically
    outward so the steep temperature gradient near the wall gets fine cells while the distant
    rock stays cheap. The outermost ring is held at the undisturbed rock temperature.
    """
    n_x: int = 50                   # slices along the drift
    n_r: int = 14                   # rock rings around each slice
    outer_radius_m: float = 60.0    # outermost ring edge (held at T_rock)

    @property
    def label(self) -> str:
        return f"radial ({self.n_x} x {self.n_r} rings)"

    def build(self, drift: DriftSpec) -> CellGraph:
        """Slice the drift along its length and wrap each slice in rock rings.

        Faces run along the drift (for the water and for each ring) and outward from ring to
        ring. Face areas for the outward faces are the cylinder surface at that radius, which is
        what makes the conduction behave radially.
        """
        nx, ncol = self.n_x, self.n_r + 1
        R = drift.section.equivalent_radius_m
        if self.outer_radius_m <= R:
            raise ValueError("outer_radius_m must exceed the drift radius")
        dx = drift.length_m / nx
        node = lambda i, j: i * ncol + j

        edges = np.geomspace(R, self.outer_radius_m, self.n_r + 1)   # ring edges
        r_c = np.concatenate([[0.0], 0.5 * (edges[:-1] + edges[1:])])  # 0 = drift axis
        xs_area = np.concatenate([[np.pi * R ** 2], np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)])
        x_c = (np.arange(nx) + 0.5) * dx

        faces = FaceList()
        for i in range(nx):
            if i < nx - 1:                                   # along the drift
                for j in range(ncol):
                    faces.add(node(i, j), node(i + 1, j), dx / 2, dx / 2, xs_area[j], 0.0)
            # drift water -> first ring (the wall). Water-side distance is the
            # geometric R here the wall-film resistance is applied later.
            faces.add(node(i, 0), node(i, 1), R, r_c[1] - R, 2 * np.pi * R * dx, 90.0)
            for j in range(1, ncol - 1):                     # ring j -> ring j+1
                rf = edges[j]                                # radius of the face between them
                faces.add(node(i, j), node(i, j + 1), rf - r_c[j], r_c[j + 1] - rf,
                          2 * np.pi * rf * dx, 90.0)

        n = nx * ncol
        ii, jj = np.divmod(np.arange(n), ncol)
        centroids = np.column_stack([x_c[ii], np.zeros(n), r_c[jj]])
        # layout: the x-r rectangle each cell occupies
        y_edges = np.concatenate([[0.0], edges])
        layout = rectangle_layout(x_c[ii], 0.5 * (y_edges[jj] + y_edges[jj + 1]),
                                  np.full(n, dx / 2), 0.5 * (y_edges[jj + 1] - y_edges[jj]))
        return CellGraph(
            volumes=np.tile(xs_area * dx, nx),
            centroids=centroids,
            drift_cells=np.array([node(i, 0) for i in range(nx)]),
            far_field_cells=np.array([node(i, ncol - 1) for i in range(nx)]),
            layout=layout,
            resolves_depth=False,
            cross_axis_label="radius from drift axis [m]",
            description=self.label,
            lattice=Lattice(x_edges=np.linspace(0.0, drift.length_m, nx + 1),
                            y_edges=np.array([-R, R]), z_edges=y_edges,
                            voxel_node=np.arange(n).reshape(nx, 1, ncol),
                            axis_labels=("distance along drift [m]", "-", "radius from drift axis [m]")),
            **faces.arrays(),
        )

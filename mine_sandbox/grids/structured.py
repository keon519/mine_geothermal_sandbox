"""Box-shaped grids around the drift: 2-D (x-z slab) and 3-D (x-y-z block).

    3-D block (three_d=True)            2-D slab (three_d=False)
    x along the drift, y across,        x along the drift, z up/down only;
    z up/down. Rock on all four         one cell wide (the drift width), so
    sides of the drift.                 rock sits above and below, the side
                                        faces are insulated.

How it is built: a tensor-product grid is laid over the whole block, with
`n_core` cells across the drift's width and height and rock cells growing
geometrically away from the wall (first one `first_cell_m` thick) out to
`extent_m`, where rock is held at T_rock. The tensor cells inside the drift
are then merged into ONE water cell per slice. The rock bordering the drift
therefore stays finely resolved on every side - which matters, because heat
leaves fastest near the drift's corners.

Cells have a real elevation, so these grids accept LayeredGeology.
`StructuredDriftGrid.plane_2d()` and `.block_3d()` are readable constructors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import DriftSpec
from .base import CellGraph, FaceList, Grid, Lattice, geometric_edges, rectangle_layout


def _axis_edges(half: float, extent: float, n_side: int, first: float, n_core: int) -> np.ndarray:
    """Cell edges across one axis: a fine core spanning the drift, with rock cells growing outward
    on both sides.
    """
    right = geometric_edges(half, extent, n_side, first)
    core = np.linspace(-half, half, n_core + 1)
    return np.concatenate([-right[::-1], core[1:-1], right])


@dataclass(frozen=True)
class StructuredDriftGrid(Grid):
    """A single drift inside a box of rock, either as a full 3-D block or a 2-D vertical slab.

    Unlike the radial grid, cells here have real positions above, below and beside the drift, so
    it can place layered rock. The 2-D slab is the cheap option when you only care about rock
    above and below; the 3-D block also captures heat leaving through the side walls and
    corners.
    """
    n_x: int = 25                 # slices along the drift
    n_side: int = 10              # rock cells from the wall to the far field, per direction
    n_core: int = 9               # cells across the drift footprint (odd; 3-D corner resolution)
    extent_m: float = 40.0        # drift axis -> far-field boundary
    first_cell_m: float = 0.25    # thickness of the rock cell touching the wall
    three_d: bool = True          # False = 2-D x-z slab

    @classmethod
    def plane_2d(cls, **kw) -> "StructuredDriftGrid":
        """A 2-D slab: rock above and below the drift only, with insulated sides."""
        return cls(three_d=False, **kw)

    @classmethod
    def block_3d(cls, **kw) -> "StructuredDriftGrid":
        """A full 3-D block: rock on all four sides of the drift."""
        return cls(three_d=True, **kw)

    @property
    def label(self) -> str:
        kind = "3-D block" if self.three_d else "2-D slab"
        return f"{kind} ({self.n_x} slices, {self.n_side} cells/side)"

    def build(self, drift: DriftSpec) -> CellGraph:
        """Lay a box grid over the drift, then merge each slice's drift cells into one water cell.

        Starting from a regular grid makes it easy to keep fine rock cells hugging the wall on
        every side. Merging the drift footprint afterwards means each slice of tunnel holds a
        single well-mixed body of water, which is how the water behaves when it flows along the
        drift.
        """
        hw, hh = 0.5 * drift.section.width_m, 0.5 * drift.section.height_m
        if self.extent_m <= max(hw, hh):
            raise ValueError("extent_m must exceed the drift half-width/height")
        # odd, so a column of cells sits on the drift's centre plane (y = 0) for profiles/sections
        core = (self.n_core | 1) if self.three_d else 1     # slab: no corners to resolve
        x_e = np.linspace(0.0, drift.length_m, self.n_x + 1)
        y_e = (_axis_edges(hw, self.extent_m, self.n_side, self.first_cell_m, core)
               if self.three_d else np.array([-hw, hw]))
        z_e = _axis_edges(hh, self.extent_m, self.n_side, self.first_cell_m, core)
        dx, dy, dz = np.diff(x_e), np.diff(y_e), np.diff(z_e)
        xc, yc, zc = (0.5 * (e[:-1] + e[1:]) for e in (x_e, y_e, z_e))
        nx, ny, nz = len(dx), len(dy), len(dz)

        # tensor cells -> final nodes (drift footprint merged per slice)
        in_y = np.abs(yc) < hw
        in_z = np.abs(zc) < hh
        I, J, K = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                                   indexing="ij"))
        tensor_drift = in_y[J] & in_z[K]
        node_of = np.empty(len(I), int)
        node_of[tensor_drift] = I[tensor_drift]                       # drift nodes 0..nx-1
        rock = np.nonzero(~tensor_drift)[0]
        node_of[rock] = nx + np.arange(len(rock))
        n_nodes = nx + len(rock)
        tid = lambda i, j, k: (i * ny + j) * nz + k

        # faces: tensor faces mapped to nodes; drift-drift faces merged
        faces, merged = FaceList(), {}
        def face(t1, t2, d1, d2, area, ang):
            a, b = node_of[t1], node_of[t2]
            if a == b:
                return                                                # inside the drift
            if tensor_drift[t1] and tensor_drift[t2]:                 # drift slice -> next slice
                merged[(a, b)] = merged.get((a, b), 0.0) + area
                return
            faces.add(a, b, d1, d2, area, ang)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    t = tid(i, j, k)
                    if i + 1 < nx:
                        face(t, tid(i + 1, j, k), dx[i] / 2, dx[i + 1] / 2, dy[j] * dz[k], 0.0)
                    if j + 1 < ny:
                        face(t, tid(i, j + 1, k), dy[j] / 2, dy[j + 1] / 2, dx[i] * dz[k], 90.0)
                    if k + 1 < nz:
                        face(t, tid(i, j, k + 1), dz[k] / 2, dz[k + 1] / 2, dx[i] * dy[j], 90.0)
        for (a, b), area in merged.items():
            faces.add(a, b, dx[a] / 2, dx[b] / 2, area, 0.0)
        # drift-side distance on wall faces is geometric here; assembly swaps in the film

        # per-node geometry
        vol_t = dx[I] * dy[J] * dz[K]
        volumes = np.bincount(node_of, weights=vol_t, minlength=n_nodes)
        cen = np.column_stack([xc[I], yc[J], zc[K]])
        centroids = np.zeros((n_nodes, 3))
        centroids[node_of[rock]] = cen[rock]
        centroids[:nx, 0] = xc                                        # drift on its axis
        edge = (K == 0) | (K == nz - 1)
        if self.three_d:
            edge |= (J == 0) | (J == ny - 1)
        far = np.unique(node_of[edge & ~tensor_drift])

        # layout: one x-z panel per tensor y column; drift drawn in the centre panel
        panel = 2.4 * self.extent_m
        lay_x, lay_y = np.empty(n_nodes), np.empty(n_nodes)
        hw_l, hh_l = np.empty(n_nodes), np.empty(n_nodes)
        lay_x[node_of[rock]], lay_y[node_of[rock]] = xc[I[rock]], zc[K[rock]] + J[rock] * panel
        hw_l[node_of[rock]], hh_l[node_of[rock]] = dx[I[rock]] / 2, dz[K[rock]] / 2
        lay_x[:nx], lay_y[:nx] = xc, (ny // 2) * panel
        hw_l[:nx], hh_l[:nx] = dx / 2, hh
        return CellGraph(
            volumes=volumes,
            centroids=centroids,
            drift_cells=np.arange(nx),
            far_field_cells=far,
            layout=rectangle_layout(lay_x, lay_y, hw_l, hh_l),
            resolves_depth=True,
            cross_axis_label="elevation relative to drift axis [m]",
            description=self.label,
            lattice=Lattice(x_edges=x_e, y_edges=y_e, z_edges=z_e,
                            voxel_node=node_of.reshape(nx, ny, nz),
                            axis_labels=("distance along drift [m]", "across drift [m]",
                                         "elevation relative to drift axis [m]")),
            **faces.arrays(),
        )

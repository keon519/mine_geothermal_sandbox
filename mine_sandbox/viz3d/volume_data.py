"""Model geometry + temperatures -> PyVista datasets for 3-D viewing and export.

No Qt here: the live view, the ParaView export and tests all use this.

    rect      RectilinearGrid on the lattice's real (uneven) cell edges, cell data
              "T" (°C), "dT" (rise above rock) and "water" (1 in workings).
              Used for exact slices and cuts.
    image     ImageData resampled onto a uniform grid (point data "T") for
              volume rendering, which needs uniform samples.
    workings  surface of the water-filled cells.

update(field) refreshes both "T" arrays in place with one gather each, so a
live view can redraw every few frames without rebuilding anything.

Vertical exaggeration is built into the coordinates (z_scale): every dataset
is already stretched, so interactive widgets, cuts and clipping all work in
one coordinate system. `bounds` are display coordinates; `true_bounds` are
metres for axis labels. Export uses z_scale = 1.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..grids import CellGraph


class Not3DError(ValueError):
    """The grid has no real 3-D shape to show (it's axisymmetric or only one cell wide)."""
    pass


def supports_3d(graph: CellGraph) -> bool:
    """True if the grid sits on a lattice with real width, length and height, so it can be shown in
    3-D.
    """
    lat = graph.lattice
    return lat is not None and graph.resolves_depth and lat.shape[1] > 1 and lat.shape[0] > 1


def _uniform_axis(edges: np.ndarray, n: int):
    """Evenly spaced sample centres along an axis, and which (uneven) cell each one falls in."""
    span = edges[-1] - edges[0]
    centres = edges[0] + (np.arange(n) + 0.5) * span / n
    return centres, np.clip(np.searchsorted(edges, centres) - 1, 0, len(edges) - 2)


class VolumeData:
    """The model as PyVista datasets, ready for 3-D viewing and export.

    Two views of the same temperatures are kept. The exact grid follows the model's real, uneven
    cell edges and is used for slices, cuts and export. A resampled uniform image is used for
    volume rendering, which needs evenly spaced samples; its resolution is capped so it stays
    light enough to redraw while a simulation runs. Lookup tables from each dataset back to
    model cells are built once, so updating the temperatures is just two array gathers.
    """
    def __init__(self, graph: CellGraph, T_rock_C: float, z_scale: float = 1.0,
                 max_samples: int = 1_200_000):
        if not supports_3d(graph):
            raise Not3DError("the 3-D view needs a grid with real 3-D geometry "
                             "(the 3-D block grid or a mine); this one is "
                             + ("axisymmetric" if not graph.resolves_depth else "one cell wide"))
        lat = graph.lattice
        self.graph, self.lattice, self.T_rock, self.z_scale = graph, lat, T_rock_C, float(z_scale)
        x_e, y_e = lat.x_edges, lat.y_edges
        z_e = lat.z_edges * self.z_scale
        vn = lat.voxel_node
        self.true_bounds = (x_e[0], x_e[-1], y_e[0], y_e[-1], lat.z_edges[0], lat.z_edges[-1])
        self.bounds = (x_e[0], x_e[-1], y_e[0], y_e[-1], z_e[0], z_e[-1])

        # exact cells (for slicing, cutting and export)
        self.rect = pv.RectilinearGrid(x_e, y_e, z_e)
        self._rect_ids = vn.ravel(order="F")                    # VTK order: x fastest
        water = graph.is_drift[self._rect_ids].astype(np.float32)
        self.rect.cell_data["water"] = water
        self.rect.cell_data["T"] = np.full(self.rect.n_cells, T_rock_C, np.float32)
        self.rect.cell_data["dT"] = np.zeros(self.rect.n_cells, np.float32)

        # uniform samples (for volume rendering)
        spans = np.array([x_e[-1] - x_e[0], y_e[-1] - y_e[0], z_e[-1] - z_e[0]])
        finest = np.array([np.diff(e).min() for e in (x_e, y_e, z_e)])
        n = np.maximum(np.ceil(spans / finest), 2).astype(int)
        while np.prod(n) > max_samples:                          # keep it light enough to stream
            n = np.maximum((n * 0.85).astype(int), 2)
        axes = [_uniform_axis(e, k) for e, k in zip((x_e, y_e, z_e), n)]
        self.image = pv.ImageData(dimensions=tuple(int(k) for k in n), spacing=tuple(spans / n),
                                  origin=tuple(a[0][0] for a in axes))
        self._img_ids = vn[np.ix_(axes[0][1], axes[1][1], axes[2][1])].ravel(order="F")
        self.image.point_data["T"] = np.full(self.image.n_points, T_rock_C, np.float32)
        self.sample_spacing = spans / n

        # the workings as a surface
        wet = self.rect.threshold(0.5, scalars="water")
        self.workings = wet.extract_surface(algorithm="dataset_surface") if wet.n_cells else pv.PolyData()

    # data
    def update(self, field: np.ndarray) -> None:
        """Refresh both datasets with a new temperature field, in place."""
        f = np.asarray(field, np.float32)
        T = f[self._rect_ids]
        self.rect.cell_data["T"] = T
        self.rect.cell_data["dT"] = T - self.T_rock
        self.image.point_data["T"] = f[self._img_ids]

    # geometry helpers
    def centres(self):
        """Cell centres in display coordinates."""
        lat = self.lattice
        return lat.centres(lat.x_edges), lat.centres(lat.y_edges), lat.centres(lat.z_edges) * self.z_scale

    def to_display(self, xyz) -> np.ndarray:
        """Convert true (x, y, z) metres to display coordinates by applying the vertical exaggeration."""
        return np.asarray(xyz, float) * np.array([1.0, 1.0, self.z_scale])

    @staticmethod
    def default_exaggeration(graph: CellGraph) -> float:
        """Stretch z so the block's height is about 3/4 of its larger plan side."""
        lat = graph.lattice
        plan = max(lat.x_edges[-1] - lat.x_edges[0], lat.y_edges[-1] - lat.y_edges[0])
        return float(np.clip(round(0.75 * plan / (lat.z_edges[-1] - lat.z_edges[0]) * 2) / 2, 1.0, 20.0))

    def axis_slice(self, axis: str, index: int) -> pv.RectilinearGrid:
        """A flat sheet of real cells: plan layer (axis 'z'), row ('y') or column ('x')."""
        lat = self.lattice
        x_e, y_e, z_e = lat.x_edges, lat.y_edges, lat.z_edges * self.z_scale
        xc, yc, zc = self.centres()
        if axis == "z":
            sheet, ids = pv.RectilinearGrid(x_e, y_e, [zc[index]]), lat.layer(index)
        elif axis == "y":
            sheet, ids = pv.RectilinearGrid(x_e, [yc[index]], z_e), lat.row(index)
        else:
            sheet, ids = pv.RectilinearGrid([xc[index]], y_e, z_e), lat.column(index)
        sheet.field_data["ids"] = ids.ravel(order="F")
        sheet.cell_data["T"] = np.full(sheet.n_cells, self.T_rock, np.float32)
        return sheet

    @staticmethod
    def fill_sheet(sheet: pv.DataSet, field: np.ndarray) -> None:
        """Fill a slice made by `axis_slice` with temperatures from a field."""
        sheet.cell_data["T"] = np.asarray(field, np.float32)[sheet.field_data["ids"]]

    def cut(self, normal, origin) -> pv.PolyData:
        """Cross-section on any plane (display coordinates), carrying the current "T"."""
        return self.rect.slice(normal=normal, origin=origin)

    def snapshot(self, field: np.ndarray, t_d: float | None = None) -> pv.RectilinearGrid:
        """A standalone copy of the exact grid with this field (for export)."""
        self.update(field)
        out = self.rect.copy(deep=True)
        if t_d is not None:
            out.field_data["time_d"] = np.array([t_d])
        return out

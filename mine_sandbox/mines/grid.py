"""MineGrid: procedurally builds a 3-D voxel model of a whole mine from a MineSpec.

    plan (x, y):  uniform cells of `plan_cell_m` over the workings, then
                  geometrically growing margin cells out to `margin_m`
    vertical (z): one layer for the seam (z = 0 at its centre), rock layers
                  growing from `first_layer_m` above and below to the extents

In the seam layer, cells whose centre falls in open workings are water
(plus, by default, one "bridge" cell wherever two open cells touch only at a
corner, so tunnels drawn at an angle stay continuous for the flow);
everything else - pillars, barriers, roof, floor - is rock. The outer faces
of the block are held at the rock temperature. Pump In / Pump Out are the
nearest water cells to the given points, or else the two water cells
farthest apart through the workings; the main route between them is the
shortest path through water.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from ..geometry import DriftSpec
from ..grids.base import CellGraph, Grid, Lattice, geometric_edges, rectangle_layout
from .spec import MineSpec


class MineGeometryError(ValueError):
    """The mine can't be turned into a usable grid, e.g. no cells land in the workings or the pumps
    aren't connected.
    """
    pass


def bridge_diagonals(open2d: np.ndarray) -> np.ndarray:
    """Open one side cell wherever two open cells meet only at a corner.

    On a square grid, a tunnel drawn at an angle turns into a staircase of cells that only touch
    at their corners, and water can't flow through a corner. Opening one of the side cells at
    each such spot keeps the tunnel continuous. This repeats until no corner-only contacts
    remain.
    """
    o = open2d.copy()
    while True:
        a, b, c, d = o[:-1, :-1], o[1:, 1:], o[1:, :-1], o[:-1, 1:]
        diag1 = a & b & ~c & ~d                     # touching along the main diagonal
        diag2 = c & d & ~a & ~b                     # touching along the anti-diagonal
        if not (diag1.any() or diag2.any()):
            return o
        o[1:, :-1] |= diag1                         # open the cell below-right of a
        o[:-1, :-1] |= diag2                        # open the cell left of d


def _plan_edges(lo: float, hi: float, cell: float, margin: float, n_margin: int) -> np.ndarray:
    """Cell edges along one plan axis: uniform cells across the workings, then cells growing
    outward through the margin.
    """
    n = max(1, int(math.ceil((hi - lo) / cell - 1e-9)))
    core = lo + np.arange(n + 1) * cell
    if n_margin <= 0 or margin <= 0:
        return core
    grow = geometric_edges(0.0, margin, n_margin, first=cell)
    return np.concatenate([core[0] - grow[::-1][:-1], core, core[-1] + grow[1:]])


def _vertical_edges(half: float, gs) -> np.ndarray:
    """Layer edges in z: the seam as one layer, with roof and floor layers growing away from it."""
    up = half + geometric_edges(0.0, gs.extent_above_m, gs.layers_above, first=gs.first_layer_m)
    down = half + geometric_edges(0.0, gs.extent_below_m, gs.layers_below, first=gs.first_layer_m)
    return np.concatenate([-down[::-1], up])


@dataclass(frozen=True)
class MineGrid(Grid):
    """Builds a 3-D voxel model of a whole mine from a MineSpec.

    The seam layer holds the water wherever the layout is open, and everything else (pillars,
    barriers, roof and floor) is rock. Unlike the single-drift grids, the workings here come
    from the mine itself, so the DriftSpec passed to `build` is ignored.
    """
    mine: MineSpec

    @property
    def plan_cell_m(self) -> float:
        """Plan cell size: the one in the mine file, or the narrowest opening so no tunnel is lost."""
        return self.mine.grid.plan_cell_m or self.mine.layout.min_opening_m

    @property
    def label(self) -> str:
        return f"mine voxels ({self.plan_cell_m:g} m plan cells)"

    def describe_workings(self, drift: DriftSpec) -> str:
        return f"mine '{self.mine.name}': {self.mine.layout.describe()}, seam {self.mine.seam.thickness_m:g} m"

    def build(self, drift: DriftSpec) -> CellGraph:
        """Voxelise the mine and return its CellGraph.

        It lays out the plan and vertical edges, marks open seam cells as water (bridging
        diagonal tunnels if asked), places the pumps and traces the route between them, then
        connects every pair of neighbouring voxels with a face. The outer skin of the block
        becomes the far field held at rock temperature.
        """
        m, gs = self.mine, self.mine.grid
        cell = self.plan_cell_m
        if cell > m.layout.min_opening_m * 1.0001:
            warnings.warn(f"plan cell {cell:g} m is coarser than the narrowest opening "
                          f"({m.layout.min_opening_m:g} m); openings may be lost", stacklevel=2)
        xmin, ymin, xmax, ymax = m.layout.bounds()
        x_e = _plan_edges(xmin, xmax, cell, gs.margin_m, gs.margin_cells)
        y_e = _plan_edges(ymin, ymax, cell, gs.margin_m, gs.margin_cells)
        z_e = _vertical_edges(0.5 * m.seam.thickness_m, gs)
        dx, dy, dz = np.diff(x_e), np.diff(y_e), np.diff(z_e)
        xc, yc, zc = Lattice.centres(x_e), Lattice.centres(y_e), Lattice.centres(z_e)
        nx, ny, nz = len(dx), len(dy), len(dz)
        k_seam = gs.layers_below
        n = nx * ny * nz
        node = np.arange(n).reshape(nx, ny, nz)

        # ---- which plan cells are open, and which open cells the pumps can reach
        XC, YC = np.meshgrid(xc, yc, indexing="ij")
        open2d = m.layout.is_open(XC, YC)
        if gs.bridge_diagonals:
            open2d = bridge_diagonals(open2d)
        if not open2d.any():
            raise MineGeometryError(f"mine '{m.name}': no grid cell centre falls in the workings; "
                                    "use a smaller grid.plan_cell_m")
        water = np.zeros((nx, ny, nz), bool)
        water[:, :, k_seam] = open2d
        inlet2d, outlet2d, path2d = self._pumps_and_path(open2d, xc, yc)

        #  faces, vectorised per axis
        a, b, da, db, area, ang = [], [], [], [], [], []
        def axis_faces(sl_a, sl_b, d_a, d_b, ar, angle):
            a.append(node[sl_a].ravel()); b.append(node[sl_b].ravel())
            shape = node[sl_a].shape
            da.append(np.broadcast_to(d_a, shape).ravel()); db.append(np.broadcast_to(d_b, shape).ravel())
            area.append(np.broadcast_to(ar, shape).ravel()); ang.append(np.full(int(np.prod(shape)), angle))
        X, Y, Z = dx[:, None, None], dy[None, :, None], dz[None, None, :]
        axis_faces((slice(0, -1),), (slice(1, None),), X[:-1] / 2, X[1:] / 2, Y * Z, 0.0)
        axis_faces((slice(None), slice(0, -1)), (slice(None), slice(1, None)),
                   Y[:, :-1] / 2, Y[:, 1:] / 2, X * Z, 90.0)
        axis_faces((slice(None), slice(None), slice(0, -1)), (slice(None), slice(None), slice(1, None)),
                   Z[:, :, :-1] / 2, Z[:, :, 1:] / 2, X * Y, 90.0)
        cat = np.concatenate

        #  far field: the outer skin of the block, never water
        skin = np.zeros((nx, ny, nz), bool)
        skin[[0, -1], :, :] = skin[:, [0, -1], :] = skin[:, :, [0, -1]] = True
        far = node[skin & ~water]

        I, J, K = (v.ravel() for v in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"))
        panel = 1.2 * (y_e[-1] - y_e[0])          # drawing: one plan panel per z layer
        to_node = lambda ij: int(node[ij[0], ij[1], k_seam])
        return CellGraph(
            volumes=(X * Y * Z).ravel(),
            centroids=np.column_stack([xc[I], yc[J], zc[K]]),
            drift_cells=node[water],
            far_field_cells=far,
            conn_a=cat(a), conn_b=cat(b), dist_a=cat(da), dist_b=cat(db),
            face_area=cat(area), angle_deg=cat(ang),
            layout=rectangle_layout(xc[I], yc[J] + K * panel, dx[I] / 2, dy[J] / 2),
            resolves_depth=True,
            cross_axis_label="height above seam centre [m]",
            description=f"{self.label}, {nx}x{ny}x{nz}",
            inlet=to_node(inlet2d), outlet=to_node(outlet2d),
            path=np.array([to_node(ij) for ij in path2d]),
            lattice=Lattice(x_edges=x_e, y_edges=y_e, z_edges=z_e, voxel_node=node,
                            axis_labels=("x [m]", "y [m]", "height above seam centre [m]")),
        )

    # pumps
    def _pumps_and_path(self, open2d, xc, yc):
        """Pump cells (plan indices) and the shortest open route between them.

        Open cells are linked to their four neighbours. Given pump positions snap to the nearest
        open cell. Without them, Pump In goes at one end of the largest connected network and
        Pump Out at the cell farthest from it through the workings, which gives the longest flow
        path and so the most rock contact.
        """
        ii, jj = np.nonzero(open2d)
        idx = -np.ones(open2d.shape, int)
        idx[ii, jj] = np.arange(len(ii))
        # 4-neighbour adjacency between open cells
        rows, cols = [], []
        for di, dj in ((1, 0), (0, 1)):
            a = idx[: open2d.shape[0] - di, : open2d.shape[1] - dj]
            b = idx[di:, dj:]
            ok = (a >= 0) & (b >= 0)
            rows.append(a[ok]); cols.append(b[ok])
        r, c = np.concatenate(rows), np.concatenate(cols)
        adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(ii), len(ii))).tocsr()

        def nearest(xy, what):
            d2 = (xc[ii] - xy[0]) ** 2 + (yc[jj] - xy[1]) ** 2
            k = int(np.argmin(d2))
            if math.sqrt(d2[k]) > 3 * self.plan_cell_m:
                warnings.warn(f"{what} at {xy} is {math.sqrt(d2[k]):.0f} m from the nearest open "
                              "cell; using that cell", stacklevel=3)
            return k

        p = self.mine.pumps
        if p.inlet_xy is not None:
            s = nearest(p.inlet_xy, "Pump In")
        else:                                       # farthest-apart pair in the largest network
            labels, nlab = ndimage.label(open2d)
            biggest = np.argmax(np.bincount(labels[ii, jj])[1:]) + 1
            start = int(np.nonzero(labels[ii, jj] == biggest)[0][0])
            d0 = shortest_path(adj, directed=False, unweighted=True, indices=start)
            s = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
        dist, pred = shortest_path(adj, directed=False, unweighted=True, indices=s,
                                   return_predecessors=True)
        if p.outlet_xy is not None:
            t = nearest(p.outlet_xy, "Pump Out")
        else:
            t = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
        if not np.isfinite(dist[t]) or s == t:
            raise MineGeometryError(f"mine '{self.mine.name}': Pump In and Pump Out are not connected "
                                    "through open workings")
        route = [t]
        while route[-1] != s:
            route.append(int(pred[route[-1]]))
        route.reverse()
        return (ii[s], jj[s]), (ii[t], jj[t]), [(ii[k], jj[k]) for k in route]


def mine_statistics(graph: CellGraph) -> dict:
    """Headline numbers for any graph with a lattice (open area, wall area, ...)."""
    water = graph.is_drift
    wall = water[graph.conn_a] ^ water[graph.conn_b]
    out = {"cells": graph.n_cells, "water_cells": int(water.sum()),
           "water_volume_m3": float(graph.volumes[water].sum()),
           "wall_area_m2": float(graph.face_area[wall].sum()),
           "route_length_m": graph.path_length_m}
    lat = graph.lattice
    if lat is not None:
        dxy = np.outer(np.diff(lat.x_edges), np.diff(lat.y_edges))
        open_plan = np.zeros(dxy.shape, bool)
        vn = lat.voxel_node
        open_plan |= np.any(water[np.maximum(vn, 0)] & (vn >= 0), axis=2)
        out["open_area_m2"] = float(dxy[open_plan].sum())
    return out

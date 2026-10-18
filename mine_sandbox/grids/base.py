"""The contract between a grid and everything downstream of it.

A Grid is a *discretisation strategy*: it holds resolution settings only and
turns a DriftSpec into a CellGraph. A CellGraph is plain data - cells,
their volumes and positions, and the faces that connect them - with no
MODFLOW, physics or plotting in it. Everything downstream (property
assignment, the MODFLOW writer, energy accounting, plots) reads only a
CellGraph, so a new grid (2-D, 3-D, a mine map, ...) needs no changes
anywhere else.

Coordinates in `centroids`: x and y horizontal, z vertical (positive up),
with the water at z = 0 (drift axis, or the centre of the mined seam). Single
drift grids also put the drift on y = 0. Axisymmetric grids use z for
"distance from the drift axis" and set `resolves_depth = False`.

Water cells ("drift_cells") may be a single chain (one tunnel) or a whole
network (a mine). `inlet`, `outlet` and `path` say where Pump In and Pump
Out are and which cells trace the main route between them; for a single
chain they default to its ends and the chain itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..geometry import DriftSpec


@dataclass(frozen=True)
class MeshLayout:
    """2-D quadrilaterals MODFLOW requires on a DISU grid (it derives connection
    direction vectors from them). Purely a drawing of the cells; the physics
    comes from the connection data. Connected cells must have distinct centres.
    Stored as arrays so large grids stay cheap when MODFLOW is not used."""
    vertices: np.ndarray    # (nv, 2) x, y
    cells: np.ndarray       # (n, 4) vertex ids of each cell, counter-clockwise
    centres: np.ndarray     # (n, 2) drawing centre of each cell

    def __len__(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class Lattice:
    """The box lattice the cells sit on, for slice views (plan layers, sections).

    voxel_node[i, j, k] is the node occupying voxel (i, j, k), or -1 if none.
    A node may cover several voxels (e.g. a merged drift cell)."""
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray
    voxel_node: np.ndarray                      # (nx, ny, nz) int
    axis_labels: tuple = ("x [m]", "y [m]", "z [m]")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of voxels along x, y and z."""
        return self.voxel_node.shape

    @staticmethod
    def centres(edges: np.ndarray) -> np.ndarray:
        """Midpoints between consecutive edges, i.e. the voxel centres along one axis."""
        return 0.5 * (edges[:-1] + edges[1:])

    def layer(self, k: int) -> np.ndarray:
        """Plan view: (nx, ny) node ids at height index k."""
        return self.voxel_node[:, :, k]

    def row(self, j: int) -> np.ndarray:
        """Vertical section along x: (nx, nz) node ids at y index j."""
        return self.voxel_node[:, j, :]

    def column(self, i: int) -> np.ndarray:
        """Vertical cross-section: (ny, nz) node ids at x index i."""
        return self.voxel_node[i, :, :]

    def voxel_of(self, node: int) -> tuple[int, int, int]:
        """First voxel covered by a node."""
        hit = np.argwhere(self.voxel_node == node)
        if len(hit) == 0:
            raise KeyError(f"node {node} is not on the lattice")
        return tuple(int(v) for v in hit[0])


@dataclass(frozen=True)
class CellGraph:
    """A grid reduced to plain data: cells, where they are, and which faces join them.

    This is the finite-volume view of the world. Each cell has a volume and a centre; each face
    records the two cells it joins, the distance from each cell centre to the face, and the face
    area. Heat conduction across a face only needs those numbers, so solvers never need to know
    whether the cells came from rings around a tunnel or a whole mine map.

    On top of that it marks which cells hold water (and the route from Pump In to Pump Out) and
    which are far-field cells pinned at the undisturbed rock temperature, standing in for the
    infinite rock beyond the model.
    """
    volumes: np.ndarray            # (n,) m^3
    centroids: np.ndarray          # (n, 3) m, drift-centred (x, y, z)
    drift_cells: np.ndarray        # node ids of the water in the drift, ordered Pump In -> Pump Out
    far_field_cells: np.ndarray    # node ids held at the undisturbed rock temperature
    conn_a: np.ndarray             # (m,) each face listed once, a < b not required
    conn_b: np.ndarray
    dist_a: np.ndarray             # (m,) distance from centre of a to the shared face
    dist_b: np.ndarray
    face_area: np.ndarray          # (m,) m^2
    angle_deg: np.ndarray          # (m,) direction a -> b in the layout plane
    layout: MeshLayout
    resolves_depth: bool           # False for axisymmetric grids (no up/down distinction)
    cross_axis_label: str = "distance from drift axis [m]"
    description: str = ""
    inlet: int | None = None       # Pump In cell   (default: first drift cell)
    outlet: int | None = None      # Pump Out cell  (default: last drift cell)
    path: np.ndarray | None = None # main route Pump In -> Pump Out (default: drift_cells)
    lattice: Lattice | None = None # for slice views; None if cells are not on a box lattice

    #basic info
    @property
    def n_cells(self) -> int:
        """Number of cells in the grid."""
        return len(self.volumes)

    @property
    def n_faces(self) -> int:
        """Number of faces, each listed once."""
        return len(self.conn_a)

    @property
    def is_drift(self) -> np.ndarray:
        """Boolean mask marking the water cells."""
        mask = np.zeros(self.n_cells, bool)
        mask[self.drift_cells] = True
        return mask

    @property
    def inlet_cell(self) -> int:
        """Cell where the loop water enters (Pump In)."""
        return int(self.drift_cells[0] if self.inlet is None else self.inlet)

    @property
    def outlet_cell(self) -> int:
        """Cell where the loop water is drawn off (Pump Out)."""
        return int(self.drift_cells[-1] if self.outlet is None else self.outlet)

    @property
    def path_cells(self) -> np.ndarray:
        """Water cells along the main route from Pump In to Pump Out, in order."""
        return self.drift_cells if self.path is None else self.path

    @property
    def path_length_m(self) -> float:
        """Length of the main route, measured centre to centre."""
        c = self.centroids[self.path_cells]
        return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    def with_distances(self, dist_a: np.ndarray, dist_b: np.ndarray) -> "CellGraph":
        """Copy with modified centre-to-face distances (used to add the wall film)."""
        from dataclasses import replace
        return replace(self, dist_a=np.asarray(dist_a, float), dist_b=np.asarray(dist_b, float))

    # geometric queries
    def along_drift(self) -> tuple[np.ndarray, np.ndarray]:
        """(distance along the main route, node ids) of the water, Pump In -> Pump Out."""
        path = self.path_cells
        c = self.centroids[path]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
        return s + (c[0, 0] if self.path is None else 0.0), path

    def wall_profile(self, at_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """(height above the water, node ids) for the cells straight up from the
        main route at a given fraction of its length - the rock temperature profile."""
        path = self.path_cells
        p = path[min(len(path) - 1, int(at_fraction * len(path)))]
        x0, y0, z0 = self.centroids[p]
        c = self.centroids
        sel = np.nonzero(np.isclose(c[:, 0], x0) & np.isclose(c[:, 1], y0) & (c[:, 2] > z0))[0]
        sel = sel[np.argsort(c[sel, 2])]
        return c[sel, 2] - z0, sel

    def section_plane(self) -> np.ndarray:
        """Node ids in the vertical plane through the middle of the main route."""
        if self.lattice is not None:
            mid = int(self.path_cells[len(self.path_cells) // 2])
            _, j, _ = self.lattice.voxel_of(mid)
            row = self.lattice.row(j)
            return np.unique(row[row >= 0])
        return np.nonzero(np.isclose(self.centroids[:, 1], 0.0))[0]

    # checking
    def validate(self) -> None:
        """Raise if the graph breaks the contract every consumer relies on."""
        n = self.n_cells
        assert self.centroids.shape == (n, 3), "centroids must be (n, 3)"
        assert np.all(self.volumes > 0), "cell volumes must be positive"
        m = self.n_faces
        for name in ("conn_b", "dist_a", "dist_b", "face_area", "angle_deg"):
            assert len(getattr(self, name)) == m, f"{name} length != number of faces"
        assert np.all((self.conn_a >= 0) & (self.conn_a < n) & (self.conn_b >= 0) & (self.conn_b < n))
        assert np.all(self.conn_a != self.conn_b), "a cell cannot connect to itself"
        lo = np.minimum(self.conn_a, self.conn_b).astype(np.int64)
        hi = np.maximum(self.conn_a, self.conn_b).astype(np.int64)
        assert len(np.unique(lo * n + hi)) == m, "each face must be listed exactly once"
        assert np.all(self.face_area > 0) and np.all(self.dist_a > 0) and np.all(self.dist_b > 0)
        assert len(self.drift_cells) >= 2, "the drift needs at least an inlet and an outlet cell"
        assert not np.any(self.is_drift[self.far_field_cells]), "drift cells cannot be pinned"
        assert len(self.layout) == n
        water = self.is_drift
        assert water[self.inlet_cell] and water[self.outlet_cell], "pumps must sit in water cells"
        assert self.inlet_cell != self.outlet_cell, "Pump In and Pump Out must be different cells"
        path = self.path_cells
        assert path[0] == self.inlet_cell and path[-1] == self.outlet_cell, "path must run inlet -> outlet"
        assert np.all(water[path]), "the path must stay in water"
        if self.lattice is not None:
            vn = self.lattice.voxel_node
            assert vn.shape == (len(self.lattice.x_edges) - 1, len(self.lattice.y_edges) - 1,
                                len(self.lattice.z_edges) - 1), "lattice edges do not match voxel_node"
            assert vn.max() < n and vn.min() >= -1
            assert len(np.unique(vn[vn >= 0])) == n, "every node must cover at least one voxel"


class Grid(ABC):
    """A strategy for cutting space into cells.

    Drift grids hold resolution settings only and take the tunnel's shape from the DriftSpec;
    grids that carry their own workings (a mine layout) ignore it. Either way the output is a
    CellGraph, so swapping grids never touches the physics or the solvers.
    """

    @abstractmethod
    def build(self, drift: DriftSpec) -> CellGraph:
        """Cut the space around the workings into cells and return them as a CellGraph."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Short name for the grid and its resolution, for summaries and the GUI."""

    def describe_workings(self, drift: DriftSpec) -> str:
        """One-line description of the water-filled workings, for summaries."""
        return f"L={drift.length_m:.0f} m, {drift.section.label}"



# Helpers shared by grid implementations

@dataclass
class FaceList:
    """Accumulates faces while a grid is being built."""
    a: list = field(default_factory=list)
    b: list = field(default_factory=list)
    da: list = field(default_factory=list)
    db: list = field(default_factory=list)
    area: list = field(default_factory=list)
    angle: list = field(default_factory=list)

    def add(self, a, b, dist_a, dist_b, area, angle_deg):
        """Record one face: the two cells it joins, their centre-to-face distances, its area and
        direction.
        """
        self.a.append(a); self.b.append(b); self.da.append(dist_a); self.db.append(dist_b)
        self.area.append(area); self.angle.append(angle_deg)

    def arrays(self) -> dict:
        """The collected faces as the numpy arrays CellGraph expects."""
        f = lambda v, t=float: np.asarray(v, dtype=t)
        return dict(conn_a=f(self.a, int), conn_b=f(self.b, int), dist_a=f(self.da),
                    dist_b=f(self.db), face_area=f(self.area), angle_deg=f(self.angle))


def rectangle_layout(cx, cy, half_w, half_h) -> MeshLayout:
    """One independent rectangle per cell (vertices are not shared)."""
    cx, cy, hw, hh = (np.asarray(v, float) for v in (cx, cy, half_w, half_h))
    corners = np.stack([np.column_stack([cx - hw, cy - hh]), np.column_stack([cx + hw, cy - hh]),
                        np.column_stack([cx + hw, cy + hh]), np.column_stack([cx - hw, cy + hh])],
                       axis=1)                                     # (n, 4, 2)
    n = len(cx)
    return MeshLayout(vertices=corners.reshape(-1, 2), cells=np.arange(4 * n).reshape(n, 4),
                      centres=np.column_stack([cx, cy]))


def geometric_edges(start: float, stop: float, n: int, first: float | None = None) -> np.ndarray:
    """n cells from `start` to `stop`, growing geometrically.
    If `first` is given the first cell has that size; otherwise edges are log-spaced."""
    if first is None:
        return np.geomspace(start, stop, n + 1)
    span = stop - start
    if first * n >= span:                    # can't grow; fall back to uniform
        return np.linspace(start, stop, n + 1)
    lo, hi = 1.0 + 1e-12, 10.0               # solve first * (g^n - 1)/(g - 1) = span for g
    for _ in range(200):
        g = 0.5 * (lo + hi)
        if first * (g ** n - 1) / (g - 1) > span:
            hi = g
        else:
            lo = g
    sizes = first * g ** np.arange(n)
    edges = start + np.concatenate([[0.0], np.cumsum(sizes)])
    edges[-1] = stop
    return edges

"""Plan-view layouts of mine workings (what is mined out, seen from above).

Every layout answers one question - is the point (x, y) open? - plus its
bounding box and its narrowest opening (used to warn when the grid is too
coarse). Layouts are registered by a type name so mine files can name them:

    room_and_pillar   parametric panel: parallel entries cut by crosscuts
    polylines         traced tunnel centrelines with a width
    polygon           any traced outline (open area, or a sealed/collapsed zone)
    rectangle         axis-aligned or rotated box
    composite         union of parts minus holes (mains + panels - roof falls)

Add a layout type by subclassing Layout and decorating it with
@register_layout("name"); the file reader picks it up automatically.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

LAYOUT_TYPES: dict[str, type["Layout"]] = {}


def register_layout(name: str):
    """Class decorator that registers a layout under the type name mine files use for it."""
    def deco(cls):
        cls.type_name = name
        LAYOUT_TYPES[name] = cls
        return cls
    return deco


class Layout(ABC):
    """A plan view of mine workings: which points (x, y) are mined out and full of water.

    Grids only ever ask a layout whether points are open, how big it is, and how narrow its
    tightest opening is. That's enough to voxelise any shape, so a parametric panel, a traced
    map and a combination of the two all work the same way. Each layout also knows how to write
    itself to and read itself from a mine file.
    """
    type_name: ClassVar[str] = ""

    @abstractmethod
    def is_open(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean array: True where the point lies in mined-out (water-filled) space."""

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the open area."""

    @property
    @abstractmethod
    def min_opening_m(self) -> float:
        """Narrowest opening; the grid's plan cell should not be coarser than this."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for summaries and the GUI."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Plain-data form for mine files (must round-trip through from_dict)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict, read_layout) -> "Layout":
        """Build from file data. `read_layout(d)` parses nested layouts."""


def _rotate(x, y, origin, angle_deg):
    """World -> local coordinates (rotate by -angle about origin)."""
    a = math.radians(angle_deg)
    dx, dy = x - origin[0], y - origin[1]
    return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)


def _rotated_bounds(corners_local, origin, angle_deg):
    """Bounding box in world coordinates of a shape given by its corners in a rotated local frame."""
    a = math.radians(angle_deg)
    pts = [(origin[0] + u * math.cos(a) - v * math.sin(a), origin[1] + u * math.sin(a) + v * math.cos(a))
           for u, v in corners_local]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)



@register_layout("room_and_pillar")
@dataclass(frozen=True)
class RoomAndPillar(Layout):
    """A panel of `n_entries` parallel entries (along local u) joined by crosscuts,
    leaving `n_pillars_along` x (n_entries - 1) rectangular pillars.

        v ^   ═══╦════╦════╦═══   entry
          |   pil║pil ║pil ║      <- pillars between entries and crosscuts
          |   ═══╬════╬════╬═══   entry
          +-----------------> u (rotated by rotation_deg about origin)
    """
    origin: tuple = (0.0, 0.0)
    rotation_deg: float = 0.0
    n_entries: int = 5
    n_pillars_along: int = 10
    entry_width_m: float = 6.0
    crosscut_width_m: float = 6.0
    pillar_length_m: float = 18.0      # along the entries
    pillar_width_m: float = 18.0       # across the entries

    def __post_init__(self):
        if self.n_entries < 1 or self.n_pillars_along < 0:
            raise ValueError("n_entries >= 1 and n_pillars_along >= 0 required")
        for name in ("entry_width_m", "crosscut_width_m", "pillar_length_m", "pillar_width_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def length_m(self) -> float:
        """Panel length along the entries, from the first crosscut to the last."""
        return (self.n_pillars_along + 1) * self.crosscut_width_m + self.n_pillars_along * self.pillar_length_m

    @property
    def width_m(self) -> float:
        """Panel width across the entries."""
        return self.n_entries * self.entry_width_m + (self.n_entries - 1) * self.pillar_width_m

    def is_open(self, x, y):
        """A point is open if it falls in an entry or a crosscut.

        Entries and crosscuts repeat at a fixed pitch (opening plus pillar), so taking the local
        coordinate modulo the pitch says whether a point is in the opening part of its repeat or
        in the pillar part.
        """
        u, v = _rotate(np.asarray(x, float), np.asarray(y, float), self.origin, self.rotation_deg)
        inside = (u >= 0) & (u <= self.length_m) & (v >= 0) & (v <= self.width_m)
        in_entry = np.mod(v, self.entry_width_m + self.pillar_width_m) < self.entry_width_m
        in_cut = np.mod(u, self.crosscut_width_m + self.pillar_length_m) < self.crosscut_width_m
        return inside & (in_entry | in_cut)

    def bounds(self):
        L, W = self.length_m, self.width_m
        return _rotated_bounds([(0, 0), (L, 0), (L, W), (0, W)], self.origin, self.rotation_deg)

    @property
    def min_opening_m(self) -> float:
        return min(self.entry_width_m, self.crosscut_width_m)

    @property
    def nominal_open_area_m2(self) -> float:
        """Mined-out area on paper: the panel's footprint minus its pillars."""
        pillars = self.n_pillars_along * (self.n_entries - 1) * self.pillar_length_m * self.pillar_width_m
        return self.length_m * self.width_m - pillars

    def describe(self) -> str:
        return (f"room-and-pillar {self.n_entries} entries x {self.n_pillars_along} pillars "
                f"({self.length_m:.0f} x {self.width_m:.0f} m)")

    def to_dict(self):
        return {"type": self.type_name, "origin": list(self.origin), "rotation_deg": self.rotation_deg,
                "n_entries": self.n_entries, "n_pillars_along": self.n_pillars_along,
                "entry_width_m": self.entry_width_m, "crosscut_width_m": self.crosscut_width_m,
                "pillar_length_m": self.pillar_length_m, "pillar_width_m": self.pillar_width_m}

    @classmethod
    def from_dict(cls, d, read_layout):
        d = dict(d)
        d["origin"] = tuple(d.get("origin", (0.0, 0.0)))
        if "crosscut_width_m" not in d and "entry_width_m" in d:
            d["crosscut_width_m"] = d["entry_width_m"]
        return cls(**d)


@dataclass(frozen=True)
class TracedDrift:
    """One traced tunnel: a centreline through two or more points, and its width."""
    points: tuple            # ((x, y), ...) centreline, at least two points
    width_m: float

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("a traced drift needs at least two points")
        if self.width_m <= 0:
            raise ValueError("width_m must be positive")


@register_layout("polylines")
@dataclass(frozen=True)
class Polylines(Layout):
    """Tunnels traced off a mine map: centrelines plus a width each."""
    drifts: tuple = ()

    def is_open(self, x, y):
        """A point is open if it lies within half a tunnel's width of any segment of that tunnel's
        centreline.
        """
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape, bool)
        for d in self.drifts:
            r2 = (0.5 * d.width_m) ** 2
            p = np.asarray(d.points, float)
            for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
                sx, sy = x1 - x0, y1 - y0
                L2 = sx * sx + sy * sy
                t = np.clip(((x - x0) * sx + (y - y0) * sy) / L2, 0.0, 1.0) if L2 > 0 else 0.0
                out |= (x - x0 - t * sx) ** 2 + (y - y0 - t * sy) ** 2 <= r2
        return out

    def bounds(self):
        pts = np.concatenate([np.asarray(d.points, float) for d in self.drifts])
        pad = max(d.width_m for d in self.drifts) / 2
        return pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad

    @property
    def min_opening_m(self) -> float:
        return min(d.width_m for d in self.drifts)

    def describe(self) -> str:
        total = sum(np.sum(np.linalg.norm(np.diff(np.asarray(d.points, float), axis=0), axis=1))
                    for d in self.drifts)
        return f"{len(self.drifts)} traced drifts, {total:.0f} m of centreline"

    def to_dict(self):
        return {"type": self.type_name,
                "drifts": [{"points": [list(p) for p in d.points], "width_m": d.width_m} for d in self.drifts]}

    @classmethod
    def from_dict(cls, d, read_layout):
        if not d.get("drifts"):
            raise ValueError("'drifts' must list at least one traced drift")
        return cls(drifts=tuple(TracedDrift(points=tuple(tuple(map(float, p)) for p in item["points"]),
                                            width_m=float(item["width_m"])) for item in d["drifts"]))



@register_layout("polygon")
@dataclass(frozen=True)
class Polygon(Layout):
    """Any closed outline (x, y) - a traced room, a gob area, or a sealed zone."""
    points: tuple = ()

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError("a polygon needs at least three points")

    def is_open(self, x, y):
        from matplotlib.path import Path
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        pts = np.column_stack([x.ravel(), y.ravel()])
        return Path(np.asarray(self.points, float)).contains_points(pts).reshape(x.shape)

    def bounds(self):
        p = np.asarray(self.points, float)
        return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()

    @property
    def min_opening_m(self) -> float:
        """The smaller side of the bounding box, a rough stand-in for the narrowest opening."""
        xmin, ymin, xmax, ymax = self.bounds()
        return min(xmax - xmin, ymax - ymin)

    def describe(self) -> str:
        return f"polygon ({len(self.points)} vertices)"

    def to_dict(self):
        return {"type": self.type_name, "points": [list(p) for p in self.points]}

    @classmethod
    def from_dict(cls, d, read_layout):
        return cls(points=tuple(tuple(map(float, p)) for p in d["points"]))


@register_layout("rectangle")
@dataclass(frozen=True)
class Rectangle(Layout):
    """A rectangular area, optionally rotated about its origin. Often used to mark a closed-off
    zone in a composite.
    """
    origin: tuple = (0.0, 0.0)
    length_m: float = 10.0
    width_m: float = 10.0
    rotation_deg: float = 0.0

    def is_open(self, x, y):
        u, v = _rotate(np.asarray(x, float), np.asarray(y, float), self.origin, self.rotation_deg)
        return (u >= 0) & (u <= self.length_m) & (v >= 0) & (v <= self.width_m)

    def bounds(self):
        L, W = self.length_m, self.width_m
        return _rotated_bounds([(0, 0), (L, 0), (L, W), (0, W)], self.origin, self.rotation_deg)

    @property
    def min_opening_m(self) -> float:
        return min(self.length_m, self.width_m)

    def describe(self) -> str:
        return f"rectangle {self.length_m:.0f} x {self.width_m:.0f} m"

    def to_dict(self):
        return {"type": self.type_name, "origin": list(self.origin), "length_m": self.length_m,
                "width_m": self.width_m, "rotation_deg": self.rotation_deg}

    @classmethod
    def from_dict(cls, d, read_layout):
        d = dict(d)
        d["origin"] = tuple(d.get("origin", (0.0, 0.0)))
        return cls(**d)



@register_layout("composite")
@dataclass(frozen=True)
class Composite(Layout):
    """Union of `parts` minus `holes` (roof falls, seals, unmined barriers)."""
    parts: tuple = ()
    holes: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not self.parts:
            raise ValueError("a composite layout needs at least one part")

    def is_open(self, x, y):
        """Open where any part is open, unless a hole covers it."""
        out = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, bool)
        for p in self.parts:
            out |= p.is_open(x, y)
        for h in self.holes:
            out &= ~h.is_open(x, y)
        return out

    def bounds(self):
        b = np.array([p.bounds() for p in self.parts])
        return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()

    @property
    def min_opening_m(self) -> float:
        return min(p.min_opening_m for p in self.parts)

    def describe(self) -> str:
        kinds = {}
        for p in self.parts:
            kinds[p.type_name] = kinds.get(p.type_name, 0) + 1
        txt = ", ".join(f"{n} {k}" for k, n in kinds.items())
        return f"composite ({txt}" + (f", {len(self.holes)} closed zones)" if self.holes else ")")

    def to_dict(self):
        return {"type": self.type_name, "parts": [p.to_dict() for p in self.parts],
                "holes": [h.to_dict() for h in self.holes]}

    @classmethod
    def from_dict(cls, d, read_layout):
        return cls(parts=tuple(read_layout(p) for p in d["parts"]),
                   holes=tuple(read_layout(h) for h in d.get("holes", [])))

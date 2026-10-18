"""Which rock sits where around the drift.

Coordinates are drift-centred: x runs along the drift from Pump In, y is
horizontal across it, z is vertical (positive up), and the drift axis is at
y = z = 0.

Every Geology answers one question - "what rock is at these points?" - and
declares whether its answer depends on position (`is_uniform`). Grids that
cannot tell up from sideways (the axisymmetric RadialDriftGrid) refuse a
non-uniform geology instead of silently mis-assigning it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .materials import Rock, get_rock


class Geology(ABC):
    """Answers "what rock is here?" for any set of points around the workings.

    Grids hand over their cell centres and get back one Rock per cell. Keeping this behind an
    interface means a grid never needs to know whether it is looking at a single rock type,
    horizontal beds, or (in future) something read from a geological model.
    """
    @abstractmethod
    def rocks_at(self, centroids: np.ndarray) -> list[Rock]:
        """Rock for each point; `centroids` has shape (n, 3) = (x, y, z)."""

    @property
    @abstractmethod
    def is_uniform(self) -> bool:
        """True if the same rock is everywhere."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Short description of the geology for summaries and plot titles."""


@dataclass(frozen=True)
class UniformGeology(Geology):
    """The same rock everywhere: the simplest case, and the only one an axisymmetric grid can
    represent.
    """
    rock: Rock

    def rocks_at(self, centroids):
        return [self.rock] * len(centroids)

    @property
    def is_uniform(self) -> bool:
        return True

    @property
    def label(self) -> str:
        return self.rock.name


@dataclass(frozen=True)
class Layer:
    """Horizontal bed between z_bottom and z_top (metres, relative to the drift axis)."""
    z_bottom: float
    z_top: float
    rock: Rock

    def contains(self, z: np.ndarray) -> np.ndarray:
        """Mask of which heights fall inside this bed (bottom inclusive, top exclusive)."""
        return (z >= self.z_bottom) & (z < self.z_top)


@dataclass(frozen=True)
class LayeredGeology(Geology):
    """Horizontal beds over a background rock. Later layers win where they overlap.

    Example - a drift driven in a 2 m coal seam, shale around it, sandstone above 4 m:
        LayeredGeology(background=ROCKS["shale"],
                       layers=(Layer(-1, 1, ROCKS["coal"]), Layer(4, 1e9, ROCKS["sandstone"])))
    """
    background: Rock
    layers: tuple[Layer, ...] = field(default_factory=tuple)

    def rocks_at(self, centroids):
        """Start every point in the background rock, then let each bed claim the points inside its
        height band.
        """
        z = np.asarray(centroids)[:, 2]
        out = [self.background] * len(z)
        for layer in self.layers:
            for i in np.nonzero(layer.contains(z))[0]:
                out[i] = layer.rock
        return out

    @property
    def is_uniform(self) -> bool:
        """True when every bed happens to be the background rock, so the layering changes nothing."""
        return all(l.rock == self.background for l in self.layers)

    @property
    def label(self) -> str:
        names = [self.background.name] + [l.rock.name for l in self.layers]
        return "layered(" + ", ".join(dict.fromkeys(names)) + ")"


def as_geology(value: "Geology | Rock | str | tuple") -> Geology:
    """Let callers pass a rock name or Rock wherever a Geology is expected."""
    return value if isinstance(value, Geology) else UniformGeology(get_rock(value))

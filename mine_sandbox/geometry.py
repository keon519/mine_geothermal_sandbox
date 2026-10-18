"""Shape of the drift (the flooded tunnel the loop water runs through).

A drift is a length plus a cross-section. Cross-sections share one small
interface (area, perimeter, width, height, equivalent radius) so every grid
can ask for the numbers it needs without knowing which shape it was given.
Add a new shape (arched roof, horseshoe, ...) by subclassing CrossSection;
nothing else has to change.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class CrossSection(ABC):
    """The shape of a drift seen end-on.

    Grids need different things from the same shape: the radial grid only cares about an
    equivalent radius, while box-shaped grids want a width and height, and the wall film needs
    the wetted perimeter. Subclasses provide the basic measurements and this base class derives
    the rest, so any grid can work with any shape.
    """
    @property
    @abstractmethod
    def area_m2(self) -> float:
        """Open area of the section, which together with porosity sets how much water the drift holds."""

    @property
    @abstractmethod
    def perimeter_m(self) -> float:
        """Length of wetted wall around the section, i.e. the surface heat crosses into the rock."""

    @property
    @abstractmethod
    def width_m(self) -> float:
        """Horizontal extent used by grids that resolve width (y)."""

    @property
    @abstractmethod
    def height_m(self) -> float:
        """Vertical extent used by grids that resolve height (z)."""

    @property
    def label(self) -> str:
        """Short human-readable size, used in summaries and plot legends."""
        return f"{self.width_m:.1f}x{self.height_m:.1f} m"

    @property
    def equivalent_radius_m(self) -> float:
        """Radius of the circle with the same area (used by axisymmetric grids)."""
        return math.sqrt(self.area_m2 / math.pi)


@dataclass(frozen=True)
class CircularSection(CrossSection):
    """A round tunnel of the given diameter.

    Grids that build boxes stand it in for a square of equal area, so the amount of water and
    the rate it flushes through stay the same.
    """
    diameter_m: float

    @property
    def area_m2(self) -> float:
        return math.pi * (0.5 * self.diameter_m) ** 2

    @property
    def perimeter_m(self) -> float:
        return math.pi * self.diameter_m

    @property
    def label(self) -> str:
        return f"D={self.diameter_m:.1f} m"

    # Box-shaped grids represent a circle by the square of equal area.
    @property
    def width_m(self) -> float:
        return math.sqrt(self.area_m2)

    @property
    def height_m(self) -> float:
        return math.sqrt(self.area_m2)


@dataclass(frozen=True)
class RectangularSection(CrossSection):
    """A box-shaped tunnel, the usual shape of a room-and-pillar coal entry."""
    width: float
    height: float

    @property
    def area_m2(self) -> float:
        return self.width * self.height

    @property
    def perimeter_m(self) -> float:
        return 2.0 * (self.width + self.height)

    @property
    def width_m(self) -> float:
        return self.width

    @property
    def height_m(self) -> float:
        return self.height


@dataclass(frozen=True)
class DriftSpec:
    """A single drift: how long it is, what shape it is, and how open it is.

    Porosity lets you model a tunnel that has partly collapsed or been backfilled with rubble:
    the water only has a fraction of the cross-section to move through, so it flows faster and
    holds less heat.
    """
    length_m: float = 500.0
    section: CrossSection = CircularSection(4.0)
    porosity: float = 1.0   # 1.0 = open flooded drift; < 1 if partly rubble-filled

    @property
    def flow_area_m2(self) -> float:
        """Cross-sectional area actually available to the water (area times porosity)."""
        return self.section.area_m2 * self.porosity

"""Thermal properties of rock and water.

Pure data, no behaviour beyond derived quantities. Values are round,
textbook-typical numbers for shallow rock; replace them with site data
(well logs, lab tests) when you have it.

    k   : thermal conductivity  [W / (m K)]
    rho : density               [kg / m^3]
    cp  : specific heat         [J / (kg K)]
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rock:
    """A material described by the three numbers heat conduction needs.

    Conductivity (k) sets how easily heat moves through the material, while density and specific
    heat together set how much heat it can soak up per degree. Water is modelled as a Rock too,
    so the same arithmetic applies to every cell in the model whether it holds rock or mine
    water.
    """
    name: str
    k: float
    rho: float
    cp: float

    @property
    def volumetric_heat_capacity(self) -> float:
        """rho * cp  [J / (m^3 K)] - heat stored per cubic metre per degree."""
        return self.rho * self.cp

    @property
    def diffusivity(self) -> float:
        """k / (rho cp)  [m^2 / s] - how fast a temperature change spreads."""
        return self.k / self.volumetric_heat_capacity


WATER = Rock("water", k=0.6, rho=1000.0, cp=4184.0)

# Appalachian coal-measure lithologies first, then generic end-members.
ROCKS: dict[str, Rock] = {
    "sandstone": Rock("sandstone", k=2.5, rho=2350.0, cp=800.0),
    "shale":     Rock("shale",     k=1.8, rho=2450.0, cp=900.0),
    "siltstone": Rock("siltstone", k=2.1, rho=2400.0, cp=850.0),
    "limestone": Rock("limestone", k=2.6, rho=2550.0, cp=850.0),
    "coal":      Rock("coal",      k=0.3, rho=1350.0, cp=1300.0),
    "granite":   Rock("granite",   k=3.0, rho=2650.0, cp=790.0),
    "basalt":    Rock("basalt",    k=1.9, rho=2900.0, cp=850.0),
    "water":     WATER,
}


def get_rock(rock: "Rock | str | tuple[float, float, float]") -> Rock:
    """Accept a Rock, a name from ROCKS, or a (k, rho, cp) tuple."""
    if isinstance(rock, Rock):
        return rock
    if isinstance(rock, str):
        try:
            return ROCKS[rock.lower()]
        except KeyError:
            raise KeyError(f"Unknown rock '{rock}'. Choose from {sorted(ROCKS)} "
                           f"or pass Rock(name, k, rho, cp).") from None
    k, rho, cp = rock
    return Rock("custom", float(k), float(rho), float(cp))

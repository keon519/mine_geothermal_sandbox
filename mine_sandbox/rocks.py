"""Thermal property library for the rock around the mineshaft.

 swap in site-specific values

    k   : thermal conductivity  [W / (m K)]
    rho : bulk density          [kg / m^3]
    cp  : specific heat         [J / (kg K)]
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Rock:
    name: str
    k: float      # W / (m K)
    rho: float    # kg / m^3
    cp: float     # J / (kg K)

    @property
    def volumetric_heat_capacity(self) -> float:
        """rho * cp  [J / (m^3 K)]"""
        return self.rho * self.cp

    @property
    def diffusivity(self) -> float:
        """k / (rho cp)  [m^2 / s] — how fast a thermal front moves into the rock."""
        return self.k / self.volumetric_heat_capacity


ROCKS = {
    "sandstone": Rock("sandstone", k=2.5, rho=2350.0, cp=800.0),
    "shale":     Rock("shale",     k=1.8, rho=2450.0, cp=900.0),
    "siltstone": Rock("siltstone", k=2.1, rho=2400.0, cp=850.0),
    "limestone": Rock("limestone", k=2.6, rho=2550.0, cp=850.0),
    "coal":      Rock("coal",      k=0.3, rho=1350.0, cp=1300.0),
    "granite":   Rock("granite",   k=3.0, rho=2650.0, cp=790.0),
    "basalt":    Rock("basalt",    k=1.9, rho=2900.0, cp=850.0),
    "water":     Rock("water",     k=0.6, rho=1000.0, cp=4184.0),
}

WATER = Rock("water", k=0.6, rho=1000.0, cp=4184.0)


def get_rock(rock) -> Rock:
    """Accept a Rock, a name from ROCKS, or a (k, rho, cp) tuple."""
    if isinstance(rock, Rock):
        return rock
    if isinstance(rock, str):
        try:
            return ROCKS[rock.lower()]
        except KeyError:
            raise KeyError(f"Unknown rock '{rock}'. Choose from {sorted(ROCKS)} "
                           f"or pass a Rock(name, k, rho, cp).") from None
    k, rho, cp = rock
    return Rock("custom", float(k), float(rho), float(cp))

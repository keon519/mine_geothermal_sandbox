"""One dataclass holding every piece of sandbox.

The four variables
    flow_m3_h, T_in_C, shaft_length_m / shaft_diameter_m, rock

"""
from dataclasses import dataclass, field, asdict
import math

from .rocks import get_rock, Rock


@dataclass
class SandboxConfig:

    flow_m3_h: float = 360.0          # water pumped through the shaft [m^3/h] (360 m3/h = 100 L/s)
    T_in_C: float = 35.0              # temperature of water entering the shaft [degC]
    shaft_length_m: float = 500.0     # distance from "Pump In" to "Pump Out" [m]
    shaft_diameter_m: float = 4.0     # equivalent circular diameter of the drift [m]
    rock: str | Rock | tuple = "sandstone"  # name in rocks.ROCKS, a Rock, or (k, rho, cp)
#reactives
    T_rock_C: float = 13.0            # undisturbed rock / groundwater temperature [degC]
    shaft_porosity: float = 1.0       # 1.0 = open flooded drift
    h_film_W_m2K: float = 800.0       # convective film coefficient water -> wall [W/m^2/K]
    rock_outer_radius_m: float = 60.0 # where rock temperature is measured at T_rock_C [m]

    n_x: int = 50                     # cells along the shaft
    n_r: int = 14                     # radial rock shells around the shaft
    sim_days: float = 365.0           # total simulated time, days
    dt_days: float = 0.5              # heat-transport time step
    adv_scheme: str = "UPSTREAM"      # "UPSTREAM" (robust at any flow) or "TVD" (sharper fronts, may fail I havent tested)


    Q_load_W: float | None = None     # heat rejected into the water by the SMR side. None = fixed T_in_C.
    pump_eta_s: float = 0.75          # isentropic pump efficiency
    p_shaft_bar: float = 1.5          # pressure at shaft outlet / pump suction
    p_pump_out_bar: float = 4.0       # pump discharge pressure
    hx_pr: float = 0.95               # pressure ratio across the SMR heat exchanger
    pipe_pr: float = 0.97             # pressure ratio across the return pipe
    coupling_step_days: float = 15.0  # only for run_coupled_sequential (the API coupling exchanges every dt)
    T_max_C: float = 95.0             # stop a coupled run if the loop water gets this hot (boiling)
 
    workspace: str = "mf6_run"
    name: str = "mine1d"

    
    @property
    def rock_props(self) -> Rock:
        return get_rock(self.rock)

    @property
    def shaft_radius_m(self) -> float:
        return 0.5 * self.shaft_diameter_m

    @property
    def shaft_area_m2(self) -> float:
        return math.pi * self.shaft_radius_m ** 2

    @property
    def flow_m3_d(self) -> float:
        return self.flow_m3_h * 24.0

    @property
    def mass_flow_kg_s(self) -> float:
        return self.flow_m3_h / 3600.0 * 1000.0

    @property
    def velocity_m_d(self) -> float:
        return self.flow_m3_d / (self.shaft_area_m2 * self.shaft_porosity)

    @property
    def residence_time_d(self) -> float:
        return self.shaft_length_m / self.velocity_m_d

    def summary(self) -> str:
        r = self.rock_props
        return (
            f"{self.name}: L={self.shaft_length_m:.0f} m, D={self.shaft_diameter_m:.1f} m, "
            f"Q={self.flow_m3_h:.0f} m3/h ({self.mass_flow_kg_s:.1f} kg/s), "
            f"T_in={self.T_in_C:.1f} C, rock={r.name} (k={r.k} W/mK, rho={r.rho:.0f}, cp={r.cp:.0f}), "
            f"residence time={self.residence_time_d*24:.1f} h"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rock"] = self.rock_props.name
        return d

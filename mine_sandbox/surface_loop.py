"""TESPy model of everything *above ground*:

    shaft outlet (T_out) ──▶ Pump ──▶ SMR-side heat exchanger (+Q_load) ──▶ return pipe ──▶ shaft inlet (T_in)

TESPy solves the steady-state thermodynamics of this loop: given the water
coming back from the mine at T_out and the heat the reactor side dumps into it
(Q_load), it returns the temperature the water will have when it re-enters
the shaft, plus pump power and pressures.  MODFLOW then handles what the
mine does to that water (see mineshaft.py / couple.py).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

from tespy.networks import Network
from tespy.components import Source, Sink, Pump, SimpleHeatExchanger, Pipe
from tespy.connections import Connection

from .config import SandboxConfig

logging.getLogger("tespy").setLevel(logging.ERROR)


@dataclass
class LoopState:
    T_in_C: float        # water temperature returned to the shaft
    T_out_C: float       # water temperature received from the shaft
    m_kg_s: float
    Q_load_W: float      # heat added on the SMR side
    pump_power_W: float
    p_shaft_in_bar: float


class SurfaceLoop:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        nw = Network()
        nw.units.set_defaults(pressure="bar", pressure_difference="bar", temperature="degC",
                              mass_flow="kg/s", power="W", heat="W")
        nw.iterinfo = False
        self.nw = nw

        self.src = Source("from shaft (Pump Out)")
        self.snk = Sink("to shaft (Pump In)")
        self.pump = Pump("circulation pump")
        self.hx = SimpleHeatExchanger("SMR heat rejection")
        self.pipe = Pipe("return pipe")

        self.c_out = Connection(self.src, "out1", self.pump, "in1", label="shaft outlet")
        self.c_pump = Connection(self.pump, "out1", self.hx, "in1", label="pump discharge")
        self.c_hx = Connection(self.hx, "out1", self.pipe, "in1", label="after SMR HX")
        self.c_in = Connection(self.pipe, "out1", self.snk, "in1", label="shaft inlet")
        nw.add_conns(self.c_out, self.c_pump, self.c_hx, self.c_in)

        self.pump.set_attr(eta_s=cfg.pump_eta_s)
        self.c_pump.set_attr(p=cfg.p_pump_out_bar)
        self.hx.set_attr(pr=cfg.hx_pr)
        self.pipe.set_attr(pr=cfg.pipe_pr, Q=0.0)   # adiabatic return pipe
        self._solved_once = False

    def solve(self, T_out_C: float, Q_load_W: float | None = None,
              m_kg_s: float | None = None) -> LoopState:
        """Given the water coming out of the mine, return what goes back in."""
        cfg = self.cfg
        m = cfg.mass_flow_kg_s if m_kg_s is None else m_kg_s
        Q = cfg.Q_load_W if Q_load_W is None else Q_load_W
        self.c_out.set_attr(fluid={"water": 1}, m=m, p=cfg.p_shaft_bar, T=T_out_C)
        if Q is None:
            # fixed-inlet-temperature mode: the HX supplies whatever it takes to hit T_in_C
            self.hx.set_attr(Q=None)
            self.c_in.set_attr(T=cfg.T_in_C)
        else:
            self.c_in.set_attr(T=None)
            self.hx.set_attr(Q=Q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev = logging.root.manager.disable
            logging.disable(logging.CRITICAL)        # TESPy is chatty while iterating
            try:
                self.nw.solve("design")
            finally:
                logging.disable(prev)
        if not self.nw.converged:
            raise RuntimeError(
                f"TESPy surface loop did not converge (T_out={T_out_C:.1f} C, Q={Q}). "
                "If T_out is near boiling the mine cannot absorb this load at this pressure.")
        return LoopState(T_in_C=self.c_in.T.val, T_out_C=T_out_C, m_kg_s=m,
                         Q_load_W=self.hx.Q.val, pump_power_W=self.pump.P.val,
                         p_shaft_in_bar=self.c_in.p.val)

    def print_results(self):
        self.nw.print_results()

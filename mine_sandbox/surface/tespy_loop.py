"""TESPy model of the surface loop:

    from drift (T_out) -> pump -> SMR heat exchanger (+Q) -> return pipe -> to drift (T_in)

The network is fixed; *what is specified* is a strategy (LoadMode):
    FixedHeatLoad(Q_W)          the SMR dumps Q into the water; TESPy returns T_in
    FixedInletTemperature(T_C)  the water is heated to exactly T_C; TESPy returns Q
Add a new operating mode (load schedule, temperature-limited load, ...) by
writing another LoadMode - the loop class does not change.

TESPy is only re-solved when T_out has moved more than `resolve_tol_K`;
in between, the mode extrapolates linearly (exact for the energy balance,
tiny error in pump heating), which keeps coupled runs fast.
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from tespy.components import Pipe, Pump, SimpleHeatExchanger, Sink, Source
from tespy.connections import Connection
from tespy.networks import Network

from ..materials import WATER
from .base import InletController, LoopState

logging.getLogger("tespy").setLevel(logging.ERROR)


@dataclass(frozen=True)
class TespyLoopSpec:
    """Fixed hardware settings for the surface loop: pump efficiency, line pressures and pressure
    losses.
    """
    pump_eta_s: float = 0.75       # isentropic pump efficiency
    p_suction_bar: float = 1.5     # pressure at the drift outlet / pump suction
    p_discharge_bar: float = 4.0   # pump discharge pressure
    hx_pr: float = 0.95            # outlet/inlet pressure ratio across the SMR exchanger
    pipe_pr: float = 0.97          # outlet/inlet pressure ratio across the return pipe


@dataclass
class _Network:
    """The TESPy objects, handed to LoadMode.configure()."""
    nw: Network
    pump: Pump
    hx: SimpleHeatExchanger
    c_out: Connection      # drift outlet -> pump
    c_in: Connection       # pipe -> drift inlet


class LoadMode(ABC):
    """How the SMR heat exchanger is specified, which decides what TESPy solves for.

    The loop network has one free specification. Either the heat load is fixed and TESPy works
    out how hot the water leaves, or the leaving temperature is fixed and TESPy works out how
    much heat that takes. Each mode also knows how to cheaply update its last answer when the
    returning water temperature only shifts slightly.
    """
    @abstractmethod
    def configure(self, net: _Network) -> None:
        """Set the one free specification on the network."""

    @abstractmethod
    def extrapolate(self, state: LoopState, dT_out: float, m_cp: float) -> LoopState:
        """Cheap update when T_out moves by dT_out since the last TESPy solve."""


@dataclass(frozen=True)
class FixedHeatLoad(LoadMode):
    """The reactor dumps a fixed amount of heat into the loop, so the inlet temperature rises and
    falls with what comes back.
    """
    Q_W: float

    def configure(self, net):
        net.c_in.set_attr(T=None)
        net.hx.set_attr(Q=self.Q_W)

    def extrapolate(self, state, dT_out, m_cp):
        """With a fixed load the loop adds the same temperature rise, so the inlet shifts by
        exactly as much as the returning water did.
        """
        return replace(state, T_in_C=state.T_in_C + dT_out)       # same rise across the loop


@dataclass(frozen=True)
class FixedInletTemperature(LoadMode):
    """The loop water is always heated to the same temperature, so the heat load is what varies."""
    T_C: float

    def configure(self, net):
        net.hx.set_attr(Q=None)
        net.c_in.set_attr(T=self.T_C)

    def extrapolate(self, state, dT_out, m_cp):
        """With a fixed inlet temperature, warmer returning water needs proportionally less heat to
        reach it (m_dot * cp * dT).
        """
        return replace(state, Q_load_W=state.Q_load_W - m_cp * dT_out)


class TespySurfaceLoop(InletController):
    """The surface loop modelled with TESPy: pump, SMR heat exchanger and return pipe.

    Water from Pump Out is pumped, heated by the SMR, and sent back down through the return
    pipe. Solving the TESPy network every step would be slow, so the result is reused and
    linearly adjusted while the returning temperature stays within `resolve_tol_K` of the last
    full solve.
    """
    def __init__(self, mode: LoadMode, mass_flow_kg_s: float,
                 spec: TespyLoopSpec | None = None, resolve_tol_K: float = 0.005):
        self.mode, self.m_dot = mode, mass_flow_kg_s
        self.spec = spec or TespyLoopSpec()
        self.tol = resolve_tol_K
        self._net = self._build(self.spec)
        mode.configure(self._net)
        self._last: tuple[float, LoopState] | None = None

    @classmethod
    def for_scenario(cls, scenario, mode: LoadMode, **kw) -> "TespySurfaceLoop":
        """Build a loop with the scenario's pumping rate."""
        return cls(mode, scenario.operation.mass_flow_kg_s, **kw)

    # build
    @staticmethod
    def _build(spec: TespyLoopSpec) -> _Network:
        """Wire up the TESPy components and set the fixed hardware specs; the load mode sets the
        last free one.
        """
        nw = Network()
        nw.units.set_defaults(pressure="bar", pressure_difference="bar", temperature="degC",
                              mass_flow="kg/s", power="W", heat="W")
        nw.iterinfo = False
        src, snk = Source("from drift (Pump Out)"), Sink("to drift (Pump In)")
        pump, hx, pipe = Pump("pump"), SimpleHeatExchanger("SMR heat exchanger"), Pipe("return pipe")
        c_out = Connection(src, "out1", pump, "in1", label="drift outlet")
        c_pump = Connection(pump, "out1", hx, "in1", label="pump discharge")
        c_hx = Connection(hx, "out1", pipe, "in1", label="after SMR HX")
        c_in = Connection(pipe, "out1", snk, "in1", label="drift inlet")
        nw.add_conns(c_out, c_pump, c_hx, c_in)
        pump.set_attr(eta_s=spec.pump_eta_s)
        c_pump.set_attr(p=spec.p_discharge_bar)
        hx.set_attr(pr=spec.hx_pr)
        pipe.set_attr(pr=spec.pipe_pr, Q=0.0)
        return _Network(nw=nw, pump=pump, hx=hx, c_out=c_out, c_in=c_in)

    # solve
    def solve(self, T_out_C: float) -> LoopState:
        """Run a full TESPy solve for a given returning water temperature.

        Raises if TESPy fails to converge, which usually means the water is approaching boiling
        at the loop pressure.
        """
        n = self._net
        n.c_out.set_attr(fluid={"water": 1}, m=self.m_dot, p=self.spec.p_suction_bar, T=T_out_C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                n.nw.solve("design")
            finally:
                logging.disable(prev)
        if not n.nw.converged:
            raise RuntimeError(f"TESPy surface loop did not converge at T_out={T_out_C:.1f} C "
                               f"({self.mode}); water may be near boiling at this pressure.")
        return LoopState(T_in_C=n.c_in.T.val, Q_load_W=n.hx.Q.val, pump_power_W=n.pump.P.val)

    def inlet_state(self, T_out_C: float, t_d: float) -> LoopState:
        """Re-solve TESPy if the returning temperature has drifted past the tolerance, otherwise
        extrapolate from the last solve.
        """
        if self._last is None or abs(T_out_C - self._last[0]) > self.tol:
            self._last = (T_out_C, self.solve(T_out_C))
            return self._last[1]
        T_ref, state = self._last
        return self.mode.extrapolate(state, T_out_C - T_ref, self.m_dot * WATER.cp)

    def print_results(self) -> None:
        """Print TESPy's own table of component and connection results."""
        self._net.nw.print_results()

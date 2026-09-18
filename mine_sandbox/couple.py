"""Couple the TESPy surface loop to the MODFLOW mineshaft.

Default: **time-step coupling through the MODFLOW 6 API.
MODFLOW runs once at every heat-transport time step

    1. read T_out  (water temperature in the last shaft cell),
    2. TESPy returns the temperature the water will have
       after the pump and the SMR-side heat exchanger  -> T_in,
    3. write T_in into the WEL inflow's TEMPERATURE auxiliary variable,
    4. MODFLOW SOLVE.

Because the exchange happens every time step (default 0.5 d, shorter
than the shaft's residence time), the closed loop behaves physically with a
fixed SMR heat load the water warms until the rock can carry the load away,
and the *rate* of warming is set by the loop's thermal inertia

Fallback: `run_coupled_sequential` re-run

Two modes, chosen by cfg.Q_load_W:
    Q_load_W = None  -> "fixed T_in": the SMR side heats the water to exactly
                        cfg.T_in_C no matter what comes back. TESPy reports
                        the heat duty that implies, plus pump power.
    Q_load_W = value -> "fixed heat load": the SMR dumps a constant Q [W] into
                        the water; T_in floats. realistic case
                    
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .config import SandboxConfig
from .mineshaft import MineshaftModel, Period, ShaftResult, SEC_PER_DAY
from .rocks import WATER
from .surface_loop import SurfaceLoop, LoopState


def find_libmf6() -> str:
    """Locate the MODFLOW 6 shared library (env MF6_LIB, ./bin, ~/bin)."""
    lib = os.environ.get("MF6_LIB")
    if lib and os.path.exists(lib):
        return lib
    names = ["libmf6.so", "libmf6.dll", "libmf6.dylib"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (os.path.join(here, "bin"), os.path.expanduser("~/bin"), os.getcwd()):
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(
        "Could not find libmf6 (the MODFLOW 6 shared library). It ships in the same "
        "zip as mf6 (https://github.com/MODFLOW-ORG/executables). Put it in ./bin or "
        "set MF6_LIB, or use run_coupled_sequential().")


@dataclass
class CoupledResult:
    cfg: SandboxConfig
    times_d: np.ndarray
    T_in_C: np.ndarray
    T_out_C: np.ndarray
    heat_removed_W: np.ndarray      # heat the shaft takes out of the water each step
    Q_load_W: np.ndarray            # heat the SMR side puts in each step
    pump_power_W: np.ndarray
    x_m: np.ndarray
    r_m: np.ndarray
    temp_final: np.ndarray          # (n_x, 1+n_r)
    temp_all: np.ndarray            # (n_steps, n_x, 1+n_r)
    rock_heat_stored_J: float = 0.0
    loop_states: list = field(default_factory=list)
    runaway: bool = False           # True if the run stopped because T reached cfg.T_max_C

    @property
    def T_shaft_final(self):
        return self.temp_final[:, 0]

    def summary(self) -> str:
        note = (f"  ** RUNAWAY: loop water reached {self.cfg.T_max_C:.0f} C at t={self.times_d[-1]:.0f} d; "
                f"this mine cannot reject this load **\n") if self.runaway else ""
        return (f"{self.cfg.summary()}\n{note}"
                f"  final: T_in={self.T_in_C[-1]:.2f} C, T_out={self.T_out_C[-1]:.2f} C, "
                f"shaft removes {self.heat_removed_W[-1]/1e6:.3f} MW of the "
                f"{self.Q_load_W[-1]/1e6:.3f} MW load, pump={self.pump_power_W[-1]/1e3:.1f} kW, "
                f"rock heat stored={self.rock_heat_stored_J/1e12:.3f} TJ")


def run_coupled(cfg: SandboxConfig, verbose: bool = False, mf6_exe: str | None = None,
                libmf6: str | None = None, tespy_tol_K: float = 0.005,
                save_every: int = 1) -> CoupledResult:
    """Time-step coupled TESPy + MODFLOW run through the MODFLOW API."""
    from modflowapi import ModflowApi

    shaft = MineshaftModel(cfg, mf6_exe=mf6_exe)
    loop = SurfaceLoop(cfg)
    ws = os.path.join(cfg.workspace, "coupled")
    # single stress period; the WEL temperature is overwritten every time step
    shaft.build([Period(cfg.sim_days, cfg.T_in_C, cfg.flow_m3_d)], workspace=ws)
    shaft.sim.write_simulation(silent=True)

    api = ModflowApi(libmf6 or find_libmf6(), working_directory=ws)
    api.initialize()
    a_idm = api.get_var_address("AUXVAR_IDM", shaft.gwfname.upper(), "WEL-1")
    a_fmi = api.get_var_address("AUXVAR", shaft.gwename.upper(), "FMI-FT1")
    a_T = api.get_var_address("X", shaft.gwename.upper())

    T_out = cfg.T_rock_C            # shaft starts full of water at rock temperature
    state = loop.solve(T_out_C=T_out)
    last_solved_T_out = T_out
    times, T_in_h, T_out_h, heat_h, load_h, pump_h, temps, states = [], [], [], [], [], [], [], []
    mcp = cfg.mass_flow_kg_s * WATER.cp
    t, t_end, istep, runaway = api.get_current_time(), api.get_end_time(), 0, False
    try:
        while t < t_end:
            if T_out > cfg.T_max_C:
                runaway = True
                break
            # --- surface loop: only re-solve TESPy when T_out moved appreciably
            if abs(T_out - last_solved_T_out) > tespy_tol_K:
                state = loop.solve(T_out_C=T_out)
                last_solved_T_out = T_out
            if cfg.Q_load_W is None:
                T_in = cfg.T_in_C                                   # fixed inlet temperature
            else:
                T_in = state.T_in_C + (T_out - last_solved_T_out)   # fixed load: T_in tracks T_out 1:1
            api.get_value_ptr(a_idm)[0] = T_in
            api.prepare_time_step(api.get_time_step())
            api.get_value_ptr(a_fmi)[0] = T_in
            api.do_time_step()
            api.finalize_time_step()
            t = api.get_current_time()
            field_ = api.get_value_ptr(a_T).reshape(shaft.nx, shaft.ncol)
            T_out = float(field_[-1, 0])
            istep += 1
            if istep % save_every == 0 or t >= t_end:
                times.append(t); T_in_h.append(T_in); T_out_h.append(T_out)
                heat_h.append(mcp * (T_in - T_out))
                # fixed-T_in mode: the HX duty is whatever closes the loop's energy balance
                load_h.append(cfg.Q_load_W if cfg.Q_load_W is not None
                              else mcp * (T_in - T_out) - state.pump_power_W)
                pump_h.append(state.pump_power_W); temps.append(field_.copy()); states.append(state)
            if verbose and (istep % max(1, int(30 / cfg.dt_days)) == 0 or t >= t_end):
                print(f"t={t:7.1f} d  T_in={T_in:6.2f}  T_out={T_out:6.2f}  "
                      f"shaft removes {mcp*(T_in-T_out)/1e6:6.3f} MW  pump={state.pump_power_W/1e3:5.1f} kW")
    finally:
        api.finalize()

    temp_all = np.array(temps)
    return CoupledResult(cfg=cfg, times_d=np.array(times), T_in_C=np.array(T_in_h),
                         T_out_C=np.array(T_out_h), heat_removed_W=np.array(heat_h),
                         Q_load_W=np.array(load_h), pump_power_W=np.array(pump_h),
                         x_m=shaft.x_centres, r_m=shaft.r_centres,
                         temp_final=temp_all[-1], temp_all=temp_all,
                         rock_heat_stored_J=shaft.rock_heat_stored_J(temp_all[-1]),
                         loop_states=states, runaway=runaway)


def run_coupled_sequential(cfg: SandboxConfig, picard_iters: int = 2, verbose: bool = False,
                           mf6_exe: str | None = None) -> CoupledResult:
    """Fallback without libmf6: re-run mf6 for every coupling step."""
    shaft = MineshaftModel(cfg, mf6_exe=mf6_exe)
    loop = SurfaceLoop(cfg)
    n_steps = int(np.ceil(cfg.sim_days / cfg.coupling_step_days))
    step_len = cfg.sim_days / n_steps
    temp_field, T_out, t_off = None, cfg.T_rock_C, 0.0
    ws = os.path.join(cfg.workspace, "coupled_seq")
    times, T_in_h, T_out_h, heat_h, load_h, pump_h, temps, states = [], [], [], [], [], [], [], []
    for k in range(n_steps):
        for _ in range(picard_iters):
            state = loop.solve(T_out_C=T_out)
            res: ShaftResult = shaft.run([Period(step_len, state.T_in_C, cfg.flow_m3_d)],
                                         temp_init=temp_field, workspace=ws)
            T_new = res.T_out_C[-1]
            converged = abs(T_new - T_out) < 0.02
            T_out = T_new
            if converged:
                break
        temp_field = res.temp_final.reshape(-1)
        times.append(res.times_d + t_off); T_in_h.append(res.T_in_C); T_out_h.append(res.T_out_C)
        heat_h.append(res.heat_removed_W); temps.append(res.temp_all)
        load_h.append(np.full(len(res.times_d), state.Q_load_W))
        pump_h.append(np.full(len(res.times_d), state.pump_power_W)); states.append(state)
        t_off += step_len
        if verbose:
            print(f"step {k+1}/{n_steps} t={t_off:.1f} d T_in={state.T_in_C:.2f} T_out={T_out:.2f}")
    temp_all = np.concatenate(temps)
    return CoupledResult(cfg=cfg, times_d=np.concatenate(times), T_in_C=np.concatenate(T_in_h),
                         T_out_C=np.concatenate(T_out_h), heat_removed_W=np.concatenate(heat_h),
                         Q_load_W=np.concatenate(load_h), pump_power_W=np.concatenate(pump_h),
                         x_m=res.x_m, r_m=res.r_m, temp_final=temp_all[-1], temp_all=temp_all,
                         rock_heat_stored_J=shaft.rock_heat_stored_J(temp_all[-1]),
                         loop_states=states)

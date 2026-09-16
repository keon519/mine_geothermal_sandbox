"""Quick checks: energy balance closes, runaway guard works.  python tests_smoke.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mine_sandbox import SandboxConfig, MineshaftModel, run_coupled

cfg = SandboxConfig(sim_days=30, workspace="mf6_run/tests", name="smoke")
m = MineshaftModel(cfg); r = m.run()
dt = np.diff(np.concatenate([[0.0], r.times_d])) * 86400
E_water = (r.heat_removed_W * dt).sum()
E_rock = m.rock_heat_stored_J(r.temp_final)
print(f"energy left the water: {E_water/1e12:.4f} TJ, stored in shaft+rock: {E_rock/1e12:.4f} TJ")
assert abs(E_water - E_rock) / E_rock < 1e-3, "energy balance does not close"
assert r.T_out_C[-1] < cfg.T_in_C and r.T_out_C[-1] > cfg.T_rock_C

c = run_coupled(SandboxConfig(sim_days=120, Q_load_W=2e6, workspace="mf6_run/tests", name="runaway"))
assert c.runaway, "2 MW into a 500 m drift should run away"
print("runaway guard OK; all smoke tests passed")

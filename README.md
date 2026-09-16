# Mine-water geothermal sandbox (MODFLOW 6 + TESPy)

A small, hackable 1-D model of the loop in your sketch: hot water is pumped
into one end of a flooded mine drift, gives heat to the surrounding rock as it
travels along the drift, and is pumped back out.

```
              surface / SMR side  (TESPy)
   Pump Out ◀── pump ── SMR heat exchanger (+Q) ── return pipe ──▶ Pump In
      ▲                                                              │
      │        flooded drift, length L, diameter D   (MODFLOW 6)      ▼
      └────────────── ◀ ◀ ◀  water  ◀ ◀ ◀ ───────────────────────────┘
                     rock (k, ρ, cp) around the drift, radial conduction
```

**MODFLOW 6** (via `flopy`) does the mine: 1-D advection of heat along the
drift (GWF + GWE packages) plus radial conduction into concentric rock shells.
**TESPy** does the surface loop: the pump, the SMR-side heat exchanger and the
return pipe, giving the temperature and pump power of the water going back in.
The two are coupled every heat-transport time step through the MODFLOW 6 API.

## Install

```bash
pip install -r requirements.txt
get-modflow :python          # downloads mf6 (and libmf6) into your Python's bin dir
```

If `get-modflow` can't reach GitHub, download `linux.zip` / `win64.zip` /
`mac.zip` from https://github.com/MODFLOW-ORG/executables/releases/latest and
copy `mf6` (or `mf6.exe`) **and** `libmf6.so` / `libmf6.dll` into `./bin/`.
Alternatively set the env vars `MF6_EXE` and `MF6_LIB`.

## Run

```bash
python run_sandbox.py           # baseline + sweeps over the 4 variables + coupled runs  (~1 min)
python run_sandbox.py --quick   # 90-day versions
python tests_smoke.py           # energy-balance and runaway checks
```

Figures land in `./figures/`, a CSV of the sweep results in
`figures/sweep_summary.csv`, MODFLOW input/output files in `./mf6_run/<name>/`.

## The four variables

All live on `SandboxConfig` (`mine_sandbox/config.py`):

| what you asked to vary | field | default |
|---|---|---|
| flow in | `flow_m3_h` | 360 m³/h (100 L/s) |
| temperature in | `T_in_C` | 35 °C |
| mineshaft size | `shaft_length_m`, `shaft_diameter_m` | 500 m, 4 m |
| rock material | `rock` | `"sandstone"` — see `mine_sandbox/rocks.py` for `coal`, `shale`, `limestone`, `granite`, … or pass `(k, rho, cp)` |

Other physics knobs: `T_rock_C` (undisturbed rock temperature, ~13 °C in SW
Virginia), `h_film_W_m2K` (water→wall convection), `shaft_porosity` (<1 if the
drift is partly rubble-filled), `rock_outer_radius_m` (where rock is pinned at
`T_rock_C`).

## Two ways to use it

**MODFLOW only, fixed inlet temperature** — "if I push water at 35 °C through
this drift, what comes out and how much heat did the rock take?"

```python
from mine_sandbox import SandboxConfig, MineshaftModel
cfg = SandboxConfig(flow_m3_h=180, T_in_C=40, shaft_length_m=800, shaft_diameter_m=3, rock="shale")
res = MineshaftModel(cfg).run()
res.T_out_C[-1], res.heat_removed_W[-1]      # outlet temperature and MW rejected at the end
res.T_shaft_final, res.temp_final            # T along the drift; full (x, r) field
```

**TESPy + MODFLOW, closed loop with an SMR heat load** — "the reactor dumps
0.3 MW into this water; where does the loop temperature settle?"

```python
from mine_sandbox import SandboxConfig, run_coupled
cfg = SandboxConfig(Q_load_W=0.3e6, flow_m3_h=360, shaft_length_m=500, rock="sandstone")
r = run_coupled(cfg, verbose=True)
print(r.summary())        # T_in, T_out, heat removed vs load, pump power, TJ stored in rock
r.runaway                 # True if the water hit T_max_C: this drift cannot reject the load
```

With `Q_load_W=None` the coupled run instead holds `T_in_C` fixed and TESPy
reports the heat-exchanger duty that implies.

## What the model is (and isn't)

* Along the drift the water is treated as well mixed across the section (plug
  flow), with upstream-weighted implicit advection — stable at any flow.
* Around the drift the rock is a set of geometric-growth annular shells; heat
  conducts radially with the rock's `k`, `ρ`, `cp`. The outermost shell is held
  at `T_rock_C`. There is no along-drift conduction bottleneck to worry about —
  the rock is the bottleneck, which is why the baseline "heat removed" barely
  moves with flow but scales with drift length, ΔT and rock conductivity.
* Not included: groundwater actually flowing through the rock (regional
  MODFLOW flow field), a free surface / open basin, heat loss at the ground
  surface, multiple drifts interacting, seasonal rock temperature. All of these
  are natural next steps — the grid is built in `MineshaftModel._build_geometry`
  and `_build_connectivity`, so adding a second row of drifts or swapping the
  rock shells for a real 2-D cross-section is local surgery.
* Units inside MODFLOW are m, d, kg, J, °C (thermal conductivities are
  multiplied by 86 400 in `mineshaft.py`).

## Files

```
mine_sandbox/config.py        SandboxConfig — every knob, with derived quantities
mine_sandbox/rocks.py         rock property library
mine_sandbox/mineshaft.py     MODFLOW 6 GWF+GWE builder / runner (DISU radial grid)
mine_sandbox/surface_loop.py  TESPy pump + SMR HX + pipe
mine_sandbox/couple.py        time-step coupling through the MODFLOW API (+ file-based fallback)
mine_sandbox/plotting.py      figures
run_sandbox.py                example driver with the four sweeps
tests_smoke.py                energy balance & runaway sanity checks
```

"""MODFLOW 6 model of the flooded mineshaft: 1-D flow along the shaft, with
radial heat conduction into the surrounding rock.



Each column i is one slice dx of the shaft. Cell (i, 0) is the water in the
shaft; cells (i, 1..n_r) are rock shells around it. Every
cell is given top=1, bot=0 and *area = its true volume*, so MODFLOW's
volume is right, and each connection's HWVA is the true face
area (shell cross-section along the shaft, 2*pi*r*dx.

Flow  : GWF, steady state. WEL injects the pumped flow at (0,0) carrying
        temperature T_in as an auxiliary variable; a CHD at (n_x-1, 0) is
        "Pump Out". Rock has negligible K.
Heat  : GWE, transient. EST stores heat in water + solid, ADV carries it along
        the shaft, CND conducts it into the rock. The water->wall film
        resistance is represented by a very short CL12 on the water side of
        the first radial connection (cl12 = k_water / h_film).

Units : metres, days, kilograms, joules, degC.  W -> J/d is a factor 86400.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np
import flopy

from .config import SandboxConfig
from .rocks import WATER

SEC_PER_DAY = 86400.0

K_CONTRAST = 1.0e-9
ROCK_POROSITY = 0.01    


def find_mf6() -> str:
    """Return a path to the mf6 executable (env MF6_EXE, PATH, or ./bin)."""
    exe = os.environ.get("MF6_EXE")
    if exe and os.path.exists(exe):
        return exe
    for cand in ("mf6", "mf6.exe"):
        p = shutil.which(cand)
        if p:
            return p
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(here, "bin", "mf6"), os.path.join(here, "bin", "mf6.exe"),
                 os.path.expanduser("~/bin/mf6")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "Could not find the MODFLOW 6 executable. Run `get-modflow :python` "
        "(ships with flopy) or set the MF6_EXE environment variable.")


@dataclass
class Period:
    """One stress period: how long it lasts, what comes in at Pump In."""
    length_d: float
    T_in_C: float
    Q_m3_d: float


@dataclass
class ShaftResult:
    times_d: np.ndarray          # end of each time step [d]
    T_out_C: np.ndarray          # water temperature at Pump Out for each step
    T_in_C: np.ndarray           # inlet temperature applied during each step
    Q_m3_d: np.ndarray           # flow applied during each step
    x_m: np.ndarray              # cell centres along the shaft
    r_m: np.ndarray              # radial cell centres (index 0 = shaft water)
    temp_final: np.ndarray       # (n_x, 1+n_r) temperature field at the end
    temp_all: np.ndarray         # (n_steps, n_x, 1+n_r) full history
    heat_removed_W: np.ndarray   # rho cp Q (T_in - T_out) per step  [W]

    @property
    def heat_removed_MW(self):
        return self.heat_removed_W / 1e6

    @property
    def T_shaft_final(self):
        """Temperature along the shaft water at the end of the run."""
        return self.temp_final[:, 0]


class MineshaftModel:
    """Build, run and read one MODFLOW 6 GWF+GWE simulation of the shaft."""

    def __init__(self, cfg: SandboxConfig, mf6_exe: str | None = None):
        self.cfg = cfg
        self.exe = mf6_exe or find_mf6()
        self._build_geometry()

   
    def _build_geometry(self):
        c = self.cfg
        nx, nr = c.n_x, c.n_r
        self.nx, self.nr = nx, nr
        self.ncol = nr + 1                      # radial cells per slice (shaft + shells)
        self.nodes = nx * self.ncol
        self.dx = c.shaft_length_m / nx
        R = c.shaft_radius_m

        # radial shell edges, geometric growth from the wall to the outer radius
        edges = np.geomspace(R, c.rock_outer_radius_m, nr + 1)
        self.r_edges = edges
        r_centres = np.empty(self.ncol)
        r_centres[0] = 0.0
        r_centres[1:] = 0.5 * (edges[:-1] + edges[1:])
        self.r_centres = r_centres

        # cross-section area of each radial cell (m^2) -> volume = area*dx
        xs_area = np.empty(self.ncol)
        xs_area[0] = np.pi * R ** 2
        xs_area[1:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        self.xs_area = xs_area
        self.x_centres = (np.arange(nx) + 0.5) * self.dx

        # film "thickness" on the water side of the wall connection
        self.film_len = WATER.k / c.h_film_W_m2K

        self._build_connectivity()

    def node(self, i, j):
        return i * self.ncol + j

    def _build_connectivity(self):
        """DISU: iac, ja, ihc, cl12, hwva. All connections are flagged
        horizontal (ihc=1) with unit thickness, so HWVA is simply the face area."""
        nx, ncol, dx = self.nx, self.ncol, self.dx
        R = self.cfg.shaft_radius_m
        conns = [[] for _ in range(self.nodes)]   # (neighbour, cl12, hwva, angldegx)

        def add(a, b, cl_a, cl_b, area, ang_ab):
            
            conns[a].append((b, cl_a, area, ang_ab))
            conns[b].append((a, cl_b, area, (ang_ab + 180.0) % 360.0))

        for i in range(nx):
            
            if i < nx - 1:
                for j in range(ncol):
                    add(self.node(i, j), self.node(i + 1, j), dx / 2, dx / 2, self.xs_area[j], 0.0)
           
            add(self.node(i, 0), self.node(i, 1),
                self.film_len, self.r_centres[1] - R, 2 * np.pi * R * dx, 90.0)
            
            for j in range(1, ncol - 1):
                r_face = self.r_edges[j]
                add(self.node(i, j), self.node(i, j + 1),
                    r_face - self.r_centres[j], self.r_centres[j + 1] - r_face,
                    2 * np.pi * r_face * dx, 90.0)

        iac, ja, ihc, cl12, hwva, ang = [], [], [], [], [], []
        for n in range(self.nodes):
            nbrs = sorted(conns[n])
            iac.append(1 + len(nbrs))
            ja.append(n); ihc.append(1); cl12.append(0.0); hwva.append(0.0); ang.append(0.0)
            for b, cl, area, a in nbrs:
                ja.append(b); ihc.append(1); cl12.append(cl); hwva.append(area); ang.append(a)
        self.iac, self.ja, self.ihc, self.cl12, self.hwva, self.angldegx = iac, ja, ihc, cl12, hwva, ang
        self.nja = len(ja)
        self.area = np.tile(self.xs_area * dx, nx)   # volume per unit 
        self._build_vertices()

    def _build_vertices(self):
        """MF6 wants VERTICES/CELL2D on a DISU grid whenever GWE is attached (it
        uses them for flow-direction unit vectors). We lay the cells out as an
        x (along shaft) by y (radius) rectangle - purely cosmetic, the physics
        comes from the connection data above, but it also makes flopy plotting work."""
        nx, ncol = self.nx, self.ncol
        xe = np.arange(nx + 1) * self.dx
        ye = np.concatenate([[0.0], self.r_edges])          # y = 0 at the shaft axis
        verts, vid = [], {}
        for iy, y in enumerate(ye):
            for ix, x in enumerate(xe):
                vid[(ix, iy)] = len(verts)
                verts.append((len(verts), float(x), float(y)))
        cell2d = []
        for i in range(nx):
            for j in range(ncol):
                n = self.node(i, j)
                v = [vid[(i, j)], vid[(i + 1, j)], vid[(i + 1, j + 1)], vid[(i, j + 1)]]
                cell2d.append((n, float(self.x_centres[i]), float(0.5 * (ye[j] + ye[j + 1])), 4, *v))
        self.vertices, self.cell2d, self.nvert = verts, cell2d, len(verts)

   
    def _cell_arrays(self):
        c = self.cfg
        rock = c.rock_props
        j = np.tile(np.arange(self.ncol), self.nx)
        is_shaft = j == 0
        k_shaft = max(1.0e4, 10.0 * c.flow_m3_d * c.shaft_length_m / self.xs_area[0])
        k = np.where(is_shaft, k_shaft, k_shaft * K_CONTRAST)
        por = np.where(is_shaft, c.shaft_porosity, ROCK_POROSITY)
        kts = np.full(self.nodes, rock.k * SEC_PER_DAY)          # J/d/m/K
        cps = np.full(self.nodes, rock.cp)
        rhos = np.full(self.nodes, rock.rho)
        return k, por, kts, cps, rhos, is_shaft

    
    def build(self, periods: list[Period], temp_init: np.ndarray | None = None,
              workspace: str | None = None):
        c = self.cfg
        ws = workspace or os.path.join(c.workspace, c.name)
        os.makedirs(ws, exist_ok=True)
        gwfname, gwename = "shaftgwf", "shaftgwe"     # MF6 caps model names at 16 chars; cfg.name picks the folder
        self.gwfname, self.gwename, self.ws = gwfname, gwename, ws

        sim = flopy.mf6.MFSimulation(sim_name="mine", sim_ws=ws, exe_name=self.exe)
        perioddata = []
        for p in periods:
            nstp = max(1, int(np.ceil(p.length_d / c.dt_days)))
            perioddata.append((p.length_d, nstp, 1.0))
        flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(periods), perioddata=perioddata)

        k, por, kts, cps, rhos, is_shaft = self._cell_arrays()
        disu_kw = dict(nodes=self.nodes, nja=self.nja, top=1.0, bot=0.0, area=self.area,
                       iac=self.iac, ja=self.ja, ihc=self.ihc, cl12=self.cl12, hwva=self.hwva,
                       angldegx=self.angldegx, nvert=self.nvert,
                       vertices=self.vertices, cell2d=self.cell2d)

        
        gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)
        flopy.mf6.ModflowGwfdisu(gwf, length_units="METERS", **disu_kw)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k)
        flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        inlet, outlet = self.node(0, 0), self.node(self.nx - 1, 0)
        wel_spd = {ip: [((inlet,), p.Q_m3_d, p.T_in_C)] for ip, p in enumerate(periods)}
        flopy.mf6.ModflowGwfwel(gwf, auxiliary=["TEMPERATURE"], stress_period_data=wel_spd,
                                pname="WEL-1", save_flows=True)
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: [((outlet,), 0.0)]},
                                pname="CHD-1", save_flows=True)
        flopy.mf6.ModflowGwfoc(gwf, budget_filerecord=f"{gwfname}.cbc",
                               head_filerecord=f"{gwfname}.hds",
                               saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")])
        ims_gwf = flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-5,
                                       inner_dvclose=1e-6, outer_maximum=100,
                                       linear_acceleration="BICGSTAB",
                                       pname="ims_gwf", filename=f"{gwfname}.ims")
        sim.register_ims_package(ims_gwf, [gwfname])

       
        gwe = flopy.mf6.ModflowGwe(sim, modelname=gwename, save_flows=True)
        flopy.mf6.ModflowGwedisu(gwe, length_units="METERS", **disu_kw)
        strt = np.full(self.nodes, c.T_rock_C) if temp_init is None else np.asarray(temp_init, float)
        flopy.mf6.ModflowGweic(gwe, strt=strt)
        flopy.mf6.ModflowGweadv(gwe, scheme=c.adv_scheme)
        flopy.mf6.ModflowGwecnd(gwe, xt3d_off=True, alh=0.0, ath1=0.0,
                                ktw=WATER.k * SEC_PER_DAY, kts=kts)
        flopy.mf6.ModflowGweest(gwe, porosity=por, heat_capacity_water=WATER.cp,
                                density_water=WATER.rho, heat_capacity_solid=cps,
                                density_solid=rhos, save_flows=True)
        flopy.mf6.ModflowGwessm(gwe, sources=[("WEL-1", "AUX", "TEMPERATURE")])
      
        ctp = [((self.node(i, self.ncol - 1),), c.T_rock_C) for i in range(self.nx)]
        flopy.mf6.ModflowGwectp(gwe, stress_period_data={0: ctp}, pname="CTP-1")
        flopy.mf6.ModflowGweoc(gwe, budget_filerecord=f"{gwename}.cbc",
                               temperature_filerecord=f"{gwename}.ucn",
                               saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "LAST")])
        ims_gwe = flopy.mf6.ModflowIms(sim, complexity="MODERATE", outer_dvclose=1e-5,
                                       inner_dvclose=1e-6, outer_maximum=300, inner_maximum=200,
                                       linear_acceleration="BICGSTAB",
                                       pname="ims_gwe", filename=f"{gwename}.ims")
        sim.register_ims_package(ims_gwe, [gwename])
        flopy.mf6.ModflowGwfgwe(sim, exgtype="GWF6-GWE6", exgmnamea=gwfname, exgmnameb=gwename)

        self.sim, self.gwf, self.gwe, self.periods = sim, gwf, gwe, periods
        return sim

  
    def run(self, periods: list[Period] | None = None, temp_init=None,
            workspace=None, silent=True) -> ShaftResult:
        c = self.cfg
        if periods is None:
            periods = [Period(c.sim_days, c.T_in_C, c.flow_m3_d)]
        self.build(periods, temp_init=temp_init, workspace=workspace)
        self.sim.write_simulation(silent=True)
        ok, buff = self.sim.run_simulation(silent=silent)
        if not ok:
            msg = "\n".join(buff[-30:])
            lst = os.path.join(self.ws, "mfsim.lst")
            if os.path.exists(lst):
                with open(lst) as f:
                    txt = f.read()
                if "ERROR REPORT" in txt:
                    msg += "\n" + txt[txt.index("ERROR REPORT"):][:1500]
            raise RuntimeError(f"MODFLOW 6 failed in {self.ws}:\n{msg}")
        return self._read(periods)

    def _read(self, periods) -> ShaftResult:
        tobj = self.gwe.output.temperature()
        times = np.array(tobj.get_times())
        data = tobj.get_alldata()                     # (nsteps, 1, 1, nodes)
        temp_all = data.reshape(len(times), self.nx, self.ncol)
        T_out = temp_all[:, -1, 0]

        # inlet T and Q applied during each step
        T_in, Q = np.empty_like(times), np.empty_like(times)
        t0, s = 0.0, 0
        for p in periods:
            t1 = t0 + p.length_d
            m = (times > t0 + 1e-9) & (times <= t1 + 1e-9)
            T_in[m], Q[m] = p.T_in_C, p.Q_m3_d
            t0 = t1
        heat_W = WATER.rho * WATER.cp * (Q / SEC_PER_DAY) * (T_in - T_out)
        return ShaftResult(times_d=times, T_out_C=T_out, T_in_C=T_in, Q_m3_d=Q,
                           x_m=self.x_centres, r_m=self.r_centres,
                           temp_final=temp_all[-1], temp_all=temp_all,
                           heat_removed_W=heat_W)

    def rock_heat_stored_J(self, temp_field: np.ndarray) -> float:
        """Energy stored in the rock relative to T_rock (a budget check)."""
        c, rock = self.cfg, self.cfg.rock_props
        vol = (self.xs_area * self.dx)[None, :]
        dT = temp_field - c.T_rock_C
        shaft = (WATER.rho * WATER.cp * c.shaft_porosity * vol[:, 0] * dT[:, 0]).sum()
        rock_e = ((1 - ROCK_POROSITY) * rock.rho * rock.cp + ROCK_POROSITY * WATER.rho * WATER.cp)
        rockJ = (rock_e * vol[:, 1:] * dT[:, 1:]).sum()
        return shaft + rockJ

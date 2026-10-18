"""Built-in finite-volume engine: a second ThermalModel, no MODFLOW needed.

It solves the same equations with the same discretisation MODFLOW 6 uses
for this problem, on any CellGraph:

  Flow (once, steady):  sum_faces C_f (h_a - h_b) = sources,
                        C_f = A / (d_a/K_a + d_b/K_b),
                        Q injected at Pump In, head fixed at Pump Out.
                        Solved on the water network only: rock is 1e9 times
                        less permeable, so the flow it would carry (~1e-9 Q)
                        is dropped. Water pockets not connected to Pump Out
                        (e.g. sealed panels) hold still water.

  Heat (every step, backward Euler, upstream advection):
      V (θ ρw cw + (1-θ) ρs cs) (T - T_old) / dt
        = sum_faces G_f (T_b - T_a)                 conduction, G_f = A / (d_a/k_a + d_b/k_b),
                                                    k = θ k_w + (1-θ) k_s
        - sum_faces ρw cw q_f T_upstream            advection
        + ρw cw Q T_in   at Pump In                 inflow
        - ρw cw Q T      at Pump Out                outflow
      with far-field cells held at T_rock.

Flow and the time step are fixed for a run, so the heat matrix never
changes. Small grids LU-factorise it once (each step is one back-
substitution); larger ones use BiCGSTAB with a physics-based block
preconditioner (exact water network + Jacobi rock, see compute/), which
needs no big factorisation and runs on the CPU or, with CuPy, on an NVIDIA
GPU. If the GPU is missing or fails, the model drops back to the CPU and
says why (`fallback_reason`, `notices`). Tests check it against MODFLOW.

    FiniteVolumeModel(problem)                       CPU, solver chosen by size
    FiniteVolumeModel(problem, device="auto")        GPU if one works, else CPU
    FiniteVolumeModel(problem, device="cuda:0", progress=print)
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, splu

from .assembly import ThermalProblem
from .compute import CPU, ArrayBackend, BlockPreconditionedSolver, SolverError, make_backend, resolve_device
from .materials import WATER
from .model import ThermalModel
from .scenario import SEC_PER_DAY


def _harmonic(area, d_a, d_b, k_a, k_b):
    """Conductance of a face between two cells, treating each half as a resistance in series.

    Heat (or water) crossing a face has to get through half of cell a and then half of cell b.
    Adding the two resistances (distance / conductivity) and inverting gives the face
    conductance, which correctly lets a low-conductivity side dominate.
    """
    return area / (d_a / k_a + d_b / k_b)


# Linear solvers for the fixed heat matrix (strategy pattern)

class LinearSolver(ABC):
    """Something that can solve the fixed heat matrix for a new right-hand side each step.

    The matrix never changes during a run, so solvers do their expensive setup once in the
    constructor and each step is cheap. They are given the previous temperatures as a starting
    guess, which iterative methods use to converge in a handful of iterations.
    """
    xb: ArrayBackend = ArrayBackend(CPU)          # where solve_backend's arrays live
    description = ""

    @abstractmethod
    def solve(self, b: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Solve M x = b and return x as a numpy array, using `guess` as a starting point where it
        helps.
        """

    def solve_backend(self, b, guess):
        """Same as solve(), on the backend's own arrays (host arrays for CPU solvers)."""
        return self.solve(b, guess)


class DirectSolver(LinearSolver):
    """Sparse LU, factorised once. Fastest per step on small grids; memory grows quickly in 3-D."""
    description = "sparse LU on the CPU (factorised once)"

    def __init__(self, M):
        self._lu = splu(M.tocsc())

    def solve(self, b, guess):
        """One back-substitution through the stored LU factors (the guess isn't needed)."""
        return self._lu.solve(b)


class IterativeSolver(LinearSolver):
    """BiCGSTAB with an incomplete-LU preconditioner, warm-started from the last
    step. The heat-capacity term keeps it well conditioned. Slower to build than
    the block solver (the ILU factorisation), kept as a reference and fallback."""
    description = "ILU-preconditioned BiCGSTAB on the CPU"

    def __init__(self, M, rtol: float = 1e-10):
        M = M.tocsc()
        M.eliminate_zeros()
        # natural ordering + no pivoting: rows already have dominant diagonals
        ilu = spilu(M, drop_tol=1e-4, fill_factor=10, permc_spec="NATURAL", diag_pivot_thresh=0.0)
        self._M, self._P, self.rtol = M, LinearOperator(M.shape, ilu.solve), rtol

    def solve(self, b, guess):
        """Run BiCGSTAB from the warm start and raise if it fails to converge."""
        x, info = bicgstab(self._M, b, x0=guess, M=self._P, rtol=self.rtol, atol=0.0, maxiter=500)
        if info != 0:
            raise RuntimeError(f"iterative heat solve did not converge (info={info})")
        return x


class BlockSolver(LinearSolver):
    """Adapter: BlockPreconditionedSolver (compute/) behind the LinearSolver interface."""

    def __init__(self, M, water, backend: ArrayBackend | None = None, rtol: float = 1e-10):
        self.impl = BlockPreconditionedSolver(M, water, backend, rtol=rtol)
        self.xb = self.impl.xb

    @property
    def description(self) -> str:
        """Which solver and device are in use, for the GUI and logs."""
        return self.impl.description

    def solve(self, b, guess):
        return self.impl.solve(b, guess)

    def solve_backend(self, b, guess):
        return self.impl.solve_backend(b, guess)


DIRECT_LIMIT = 3_000       # cells; "auto" factorises grids up to this size, above it uses "block"
SOLVER_KINDS = ("auto", "direct", "iterative", "block")


def make_solver(M, kind: str = "auto", water: np.ndarray | None = None,
                backend: ArrayBackend | None = None) -> LinearSolver:
    """Pick and build a linear solver for the heat matrix.

    - direct: LU factorisation, fastest per step on small grids.
    - iterative: ILU-preconditioned BiCGSTAB, kept as a reference and fallback.
    - block: BiCGSTAB with the physics-based preconditioner, on the CPU or a GPU.
    - auto: direct up to DIRECT_LIMIT cells on the CPU, block otherwise.
    """
    if kind == "auto":
        kind = "direct" if M.shape[0] <= DIRECT_LIMIT and (backend is None or not backend.device.is_gpu) \
            else "block"
    if kind == "block":
        if water is None:
            raise ValueError("the block solver needs the water-cell mask")
        return BlockSolver(M, water, backend)
    if kind not in ("direct", "iterative"):
        raise ValueError(f"unknown solver '{kind}' (choose from {', '.join(SOLVER_KINDS)})")
    return {"direct": DirectSolver, "iterative": IterativeSolver}[kind](M)


def _water_network_flow(problem: ThermalProblem, Q: float):
    """Work out the steady flow of water through the workings; returns (face flows, heads).

    Water is pushed in at Pump In and drawn out at Pump Out, and it spreads through the
    connected openings like current through a network of resistors: each face's flow is its
    conductance times the head difference across it. Only the water cells are solved, since rock
    is effectively impermeable, and any pockets not connected to the pumps (such as sealed
    panels) are left with no flow at all.
    """
    g, K = problem.graph, problem.props.hydraulic_k_m_d
    n, a, b = g.n_cells, g.conn_a, g.conn_b
    water = g.is_drift
    ww = water[a] & water[b]
    wid = np.cumsum(water) - 1                        # cell -> water index
    nw = int(water.sum())
    aw, bw = wid[a[ww]], wid[b[ww]]
    C = _harmonic(g.face_area[ww], g.dist_a[ww], g.dist_b[ww], K[a[ww]], K[b[ww]])
    adj = sp.coo_matrix((np.ones(len(aw)), (aw, bw)), shape=(nw, nw))
    _, comp = connected_components(adj, directed=False)
    live = comp == comp[wid[g.outlet_cell]]          # the network the pumps are on
    li = np.cumsum(live) - 1
    keep = live[aw] & live[bw]
    ka, kb, kc = li[aw[keep]], li[bw[keep]], C[keep]
    nl = int(live.sum())
    L = sp.coo_matrix((np.r_[kc, kc, -kc, -kc], (np.r_[ka, kb, ka, kb], np.r_[ka, kb, kb, ka])),
                      shape=(nl, nl)).tocsr()
    rhs = np.zeros(nl)
    rhs[li[wid[g.inlet_cell]]] += Q
    out = li[wid[g.outlet_cell]]
    fix = np.zeros(nl)
    fix[out] = 1.0
    L = sp.diags(1.0 - fix) @ L + sp.diags(fix)       # head = 0 at Pump Out
    rhs[out] = 0.0
    h_live = splu(L.tocsc()).solve(rhs)
    heads = np.zeros(n)
    water_idx = np.nonzero(water)[0]
    heads[water_idx[live]] = h_live
    flows = np.zeros(len(a))
    idx = np.nonzero(ww)[0][keep]
    flows[idx] = kc * (heads[a[idx]] - heads[b[idx]])
    return flows, heads


ProgressFn = Callable[[str], None]       # called with a stage key: "flow", "matrix", "solver"


class FiniteVolumeModel(ThermalModel):
    """The built-in heat model: a fast finite-volume solver that matches MODFLOW.

    Setup does all the heavy lifting once. It solves the steady water flow, builds the heat
    matrix that combines heat storage, conduction between cells and heat carried along by the
    moving water, and then picks a solver and a device (CPU or GPU) for it. After that each time
    step is a single linear solve.

    If the GPU fails, at setup or mid-run, the model carries on from the same state on the CPU
    and records why in `fallback_reason` and `notices`, so a long run is never lost to a driver
    hiccup.
    """
    def __init__(self, problem: ThermalProblem, T_init: np.ndarray | None = None,
                 solver: str = "auto", device: str = "cpu", progress: ProgressFn | None = None):
        report = progress or (lambda stage: None)
        self.problem = problem
        sc, g, p = problem.scenario, problem.graph, problem.props
        self.dt = sc.numerics.dt_days
        self._end = sc.numerics.sim_days
        self._t = 0.0
        self._T_in = sc.operation.T_in_C
        self._Q = sc.operation.flow_m3_d
        self._rwcw = WATER.rho * WATER.cp                         # J / (m^3 K)
        self.notices: list[str] = []                              # solver/device changes during the run
        n = g.n_cells
        a, b = g.conn_a, g.conn_b

        #  steady flow -> face flows q (m^3/d, + means a -> b)
        report("flow")
        self.face_flow, self.heads = _water_network_flow(problem, self._Q)

        # heat matrix (J/d per K)
        report("matrix")
        k_eff = (p.porosity * WATER.k + (1 - p.porosity) * p.k_solid) * SEC_PER_DAY
        G = _harmonic(g.face_area, g.dist_a, g.dist_b, k_eff[a], k_eff[b])
        self.capacity = g.volumes * p.volumetric_heat_capacity / self.dt
        fa = self._rwcw * np.maximum(self.face_flow, 0.0)        # a -> b advective conductance
        fb = self._rwcw * np.maximum(-self.face_flow, 0.0)       # b -> a
        diag = self.capacity.copy()
        np.add.at(diag, a, G + fa)
        np.add.at(diag, b, G + fb)
        # water leaving through Pump Out = what the faces deliver to it
        out = g.outlet_cell
        q_out = (np.sum(self.face_flow[b == out]) - np.sum(self.face_flow[a == out]))
        diag[out] += self._rwcw * max(q_out, 0.0)
        rows = np.r_[np.arange(n), a, b]
        cols = np.r_[np.arange(n), b, a]
        vals = np.r_[diag, -(G + fb), -(G + fa)]
        M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        # far field: identity rows, value supplied on the right-hand side
        fixed = np.zeros(n, bool)
        fixed[g.far_field_cells] = True
        keep = sp.diags((~fixed).astype(float))
        M = (keep @ M + sp.diags(fixed.astype(float))).tocsc()
        self._M = M
        self._fixed, self._T_fixed = fixed, sc.ground.T_rock_C
        self._inlet = g.inlet_cell
        T0 = (np.full(n, sc.ground.T_rock_C) if T_init is None
              else np.asarray(T_init, float).copy())

        # solver, on the requested device (CPU fallback with a reason)
        report("solver")
        self.solver_kind = solver
        self.selection_note: str | None = None      # why "auto" picked the device it did
        self.device, self.fallback_reason = resolve_device(device)
        self._solver = None
        self._last_host: np.ndarray | None = None
        self.tuning = None
        if self.device.is_gpu and solver in ("direct", "iterative"):
            self.device, self.fallback_reason = CPU, f"the {solver} solver runs on the CPU only"
        gpu_s = None
        if self.device.is_gpu:
            try:
                self._solver = make_solver(M, "block", g.is_drift, make_backend(self.device))
                self._bind(T0)
                self.tuning = self._solver.impl.tune(self._rhs(), self._T)    # also compiles the kernels
                gpu_s = min(self.tuning.values())
            except Exception as e:  # noqa: BLE001 - out of memory, driver or kernel errors
                self.fallback_reason = f"the GPU could not be used ({type(e).__name__}: {e})"
                self.device, self._solver = CPU, None
        if self._solver is not None and str(device).lower() == "auto":
            self._pick_faster(M, T0, gpu_s)          # small models can be quicker on the CPU
        if self._solver is None:
            self._solver = make_solver(M, solver, g.is_drift)
            self._bind(T0)
            if isinstance(self._solver, BlockSolver):
                self.tuning = self._solver.impl.tune(self._rhs(), self._T)
        if self.device.is_gpu:
            self._last_host = T0.copy()               # what to resume from if the GPU dies on step 1

    def _pick_faster(self, M, T0: np.ndarray, gpu_s: float) -> None:
        """device="auto" with a working GPU: time the CPU solver on the same first step and
        keep whichever is faster on this machine."""
        gpu, gpu_label = self._solver, self.device.label
        cpu = make_solver(M, self.solver_kind, self.problem.graph.is_drift)
        b = self._xb.to_host(self._rhs())
        cpu_tuning = None
        if isinstance(cpu, BlockSolver):
            cpu_tuning = cpu.impl.tune(b, T0)
            cpu_s = min(cpu_tuning.values())
        else:
            cpu_s = math.inf
            for _ in range(2):
                t0 = time.perf_counter()
                cpu.solve(b, T0)
                cpu_s = min(cpu_s, time.perf_counter() - t0)
        timing = f"{1e3 * gpu_s:.1f} ms per solve on the GPU vs {1e3 * cpu_s:.1f} ms on the CPU"
        if cpu_s < gpu_s:
            self.selection_note = f"Auto chose the CPU for this model ({timing})"
            self.device, self._solver = CPU, cpu
            self._bind(T0)
            self.tuning = cpu_tuning
        else:
            self.selection_note = f"Auto chose {gpu_label} ({timing})"
            self._solver = gpu

    # internals
    def _bind(self, T_host: np.ndarray) -> None:
        """Put the state and right-hand-side pieces on the solver's backend."""
        xb = self._xb = self._solver.xb
        self._cap = xb.asarray(self.capacity)
        self._fixed_idx = xb.asindex(np.nonzero(self._fixed)[0])
        self._T = xb.asarray(T_host)
        self.tuning = None

    def _rhs(self):
        """Right-hand side for the next step.

        With backward Euler, the heat stored from the last step (capacity times temperature)
        drives the new solve. The incoming loop water adds its heat at Pump In, and far-field
        rows are simply set to the rock temperature to hold them fixed.
        """
        b = self._cap * self._T
        b[self._inlet] += self._rwcw * self._Q * self._T_in
        b[self._fixed_idx] = self._T_fixed
        return b

    def _fall_back(self, err: Exception, T_host: np.ndarray) -> None:
        """GPU trouble mid-run (or while tuning): carry on from the same state on the CPU."""
        was = self.solver_description
        # GPU -> the same choice on the CPU; CPU block solver -> the ILU solver
        kind = self.solver_kind if self.device.is_gpu else "iterative"
        self.device = CPU
        self.fallback_reason = f"{type(err).__name__}: {err}"
        self._solver = make_solver(self._M, kind, self.problem.graph.is_drift)
        self._bind(T_host)
        self.notices.append(f"{was} failed ({self.fallback_reason}); continuing on the CPU with "
                            f"{self.solver_description}")

    @property
    def solver_description(self) -> str:
        """Human-readable description of the solver and device currently in use."""
        return self._solver.description

    @property
    def last_iterations(self) -> int | None:
        """Iterations the block solver needed on the last step (None for other solvers)."""
        return self._solver.impl.last_iterations if isinstance(self._solver, BlockSolver) else None

    # interface
    @property
    def time_d(self) -> float:
        return self._t

    @property
    def end_time_d(self) -> float:
        return self._end

    def set_inlet_temperature(self, T_C: float) -> None:
        """Set the temperature of the water entering at Pump In for the next step."""
        self._T_in = float(T_C)

    def advance(self) -> None:
        """Take one time step, switching to the CPU and retrying the step if the GPU or block
        solver fails.
        """
        try:
            self._T = self._solver.solve_backend(self._rhs(), self._T)
        except Exception as e:  # noqa: BLE001
            recoverable = self.device.is_gpu or isinstance(e, SolverError)
            if not recoverable or not isinstance(self._solver, BlockSolver):
                raise
            self._fall_back(e, self._host_state())
            self._T = self._solver.solve_backend(self._rhs(), self._T)
        self._t += self.dt

    def _host_state(self) -> np.ndarray:
        """The state before the failed step. After a fatal CUDA error the device can't be
        read any more; the copy handed out by the last temperatures() call is the same state."""
        try:
            return np.array(self._xb.to_host(self._T), float)
        except Exception:  # noqa: BLE001
            if self._last_host is not None:
                return self._last_host.copy()
            raise

    def temperatures(self) -> np.ndarray:
        """Fresh host-side copy of every cell's temperature, also kept as a recovery point when on
        a GPU.
        """
        T = np.array(self._xb.to_host(self._T), float)      # always a fresh host copy
        if self.device.is_gpu:
            self._last_host = T
        return T

    def outlet_temperature(self) -> float:
        """Temperature at Pump Out, read directly so a GPU run doesn't copy the whole field each step."""
        return self._xb.item(self._T, self.problem.graph.outlet_cell)

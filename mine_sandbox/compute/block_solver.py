"""Block-preconditioned BiCGSTAB for the fixed heat matrix, on the CPU or a GPU.

The heat matrix has two very different parts:

  * water cells (a few hundred to a few thousand): advection-dominated. Water
    crosses tens of cells per time step, so local methods (Jacobi, plain Krylov)
    need hundreds of iterations to carry heat down the workings.
  * rock cells (everything else): conduction plus a large heat-capacity term,
    so each row is strongly diagonally dominant.

The preconditioner follows that physics (block Gauss-Seidel):

    z_water = M_ww^-1 r_water                exact: sparse LU of the small water network
    z_rock  = D_rr^-1 (r_rock - M_rw z_water)  Jacobi, seeing the water it touches
    (+ `sweeps` Jacobi sweeps on the rock)

With the previous step as the starting guess, BiCGSTAB converges to 1e-10 in
2-5 iterations on mine grids. Every operation is a sparse mat-vec or a vector
update, which is exactly what a GPU is good at; the only host round trip per
preconditioner call is the water block (a few kB). There is no big
factorisation, so building is fast too.

All arithmetic goes through an ArrayBackend, so the CPU and GPU run the same
code (and the tests exercise the GPU path's logic on numpy).
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .devices import CPU, ArrayBackend


class SolverError(RuntimeError):
    """The iterative solve broke down or did not converge."""


class BlockPreconditionedSolver:
    """BiCGSTAB with a preconditioner that splits the problem into water and rock.

    Construction does the one-off setup: it factorises the small water network on the host, and
    moves the full matrix, its inverse diagonal and the rock-to-water coupling onto the backend
    (CPU or GPU). Each solve then only needs cheap vector operations plus one small water solve
    per preconditioner call.

    `sweeps` adds extra Jacobi passes over the rock; `tune()` picks how many pay off for a given
    model.
    """
    def __init__(self, M: sp.spmatrix, water: np.ndarray, backend: ArrayBackend | None = None,
                 rtol: float = 1e-10, maxiter: int = 300, sweeps: int = 0):
        self.xb = backend or ArrayBackend(CPU)
        self.rtol, self.maxiter, self.sweeps = float(rtol), int(maxiter), int(sweeps)
        M = sp.csr_matrix(M)
        M.eliminate_zeros()
        n = M.shape[0]
        water = np.asarray(water, bool)
        d = M.diagonal()
        if np.any(d == 0):
            raise SolverError("heat matrix has a zero on its diagonal")
        widx = np.nonzero(water)[0]
        rock = ~water
        # host: exact factorisation of the (small) water network
        self._lu_w = splu(M[widx][:, widx].tocsc()) if len(widx) else None
        # device: matrix, inverse diagonal, rock<-water coupling
        C = sp.diags(rock.astype(float)) @ M[:, widx]
        C = sp.csr_matrix(C)
        C.eliminate_zeros()
        xb = self.xb
        self.n, self.n_water = n, len(widx)
        self.A = xb.csr(M)
        self.dinv = xb.asarray(1.0 / d)
        self.rock_dinv = xb.asarray(np.where(rock, 1.0 / d, 0.0))
        self.C = xb.csr(C)
        self.widx = xb.asindex(widx)
        self.last_iterations = 0
        self.total_iterations = 0
        self.solves = 0

    @property
    def description(self) -> str:
        """Which device the solver runs on and how the preconditioner is set up, for display."""
        where = "GPU" if self.xb.device.is_gpu else "CPU"
        return f"block-preconditioned BiCGSTAB on the {where} (water network exact, rock Jacobi×{1 + self.sweeps})"

    # preconditioner
    def precondition(self, r):
        """Approximately invert the heat matrix for one vector (block Gauss-Seidel).

        The water part is solved exactly, since heat moves through it too fast for local
        methods. The rock part is then estimated from its diagonal, after accounting for the
        water temperatures it touches.
        """
        xb = self.xb
        z = self.dinv * r
        if self._lu_w is not None:
            zw = xb.asarray(self._lu_w.solve(xb.to_host(r[self.widx])))
            z[self.widx] = zw
            # Rock next to the workings feels the water's new temperatures through the coupling
            # block.
            z -= self.dinv * (self.C @ zw)
        for _ in range(self.sweeps):
            z += self.rock_dinv * (r - self.A @ z)
        return z

    # BiCGSTAB
    def solve_backend(self, b, x0):
        """Solve on the backend's arrays (device arrays on the GPU). Returns x."""
        xb, xp, A, P = self.xb, self.xb.xp, self.A, self.precondition
        bnorm = xb.norm(b)
        if bnorm == 0.0:
            self.last_iterations = 0
            return xp.zeros_like(b)
        tol = self.rtol * bnorm
        x = x0.copy()
        r = b - A @ x
        if xb.norm(r) <= tol:
            self._count(0)
            return x
        rhat = r.copy()
        p = v = rho_old = alpha = omega = None
        for it in range(1, self.maxiter + 1):
            rho = xp.dot(rhat, r)
            if it == 1:
                p = r.copy()
            else:
                p = r + (rho / rho_old) * (alpha / omega) * (p - omega * v)
            ph = P(p)
            v = A @ ph
            alpha = rho / xp.dot(rhat, v)
            s = r - alpha * v
            snorm = xb.norm(s)
            if not math.isfinite(snorm):
                raise SolverError(f"BiCGSTAB broke down at iteration {it}")
            if snorm <= tol:
                self._count(it)
                return x + alpha * ph
            sh = P(s)
            t = A @ sh
            omega = xp.dot(t, s) / xp.dot(t, t)
            x = x + alpha * ph + omega * sh
            r = s - omega * t
            rnorm = xb.norm(r)
            if not math.isfinite(rnorm):
                raise SolverError(f"BiCGSTAB broke down at iteration {it}")
            if rnorm <= tol:
                self._count(it)
                return x
            rho_old = rho
        raise SolverError(f"BiCGSTAB did not reach rtol={self.rtol:g} in {self.maxiter} iterations")

    def _count(self, it: int) -> None:
        """Record how many iterations the last solve needed, for statistics and tuning."""
        self.last_iterations = it
        self.total_iterations += it
        self.solves += 1

    def solve(self, b: np.ndarray, guess: np.ndarray) -> np.ndarray:
        """Host arrays in and out (the LinearSolver interface)."""
        xb = self.xb
        return xb.to_host(self.solve_backend(xb.asarray(b), xb.asarray(guess)))

    #tuning
    def tune(self, b, x0, candidates=(0, 1), repeats: int = 2) -> dict[int, float]:
        """Try each number of rock sweeps on a trial solve and keep the cheapest.

        The choice is made on iteration counts (in mat-vec equivalents), not wall time,
        so the same model always gets the same solver and the same answer. Returns the
        best wall time per candidate; on the GPU the first repeat compiles the kernels."""
        times, cost = {}, {}
        for sweeps in candidates:
            self.sweeps = sweeps
            best = math.inf
            for _ in range(repeats):
                t0 = time.perf_counter()
                self.solve_backend(b, x0)
                self.xb.synchronize()
                best = min(best, time.perf_counter() - t0)
            times[sweeps] = best
            cost[sweeps] = max(self.last_iterations, 1) * (3.5 + 2 * sweeps)   # A-mat-vecs per iteration
        self.sweeps = min(cost, key=lambda k: (cost[k], k))
        self.total_iterations = self.solves = 0
        return times

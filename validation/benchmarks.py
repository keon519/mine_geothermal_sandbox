"""Independent reference solutions for checking the MODFLOW grids.

None of this code shares anything with mine_sandbox's solver path: it is
plain numpy/scipy, so agreement is real evidence, not a tautology. Each
function returns the heat flow into the rock per metre of drift [W/m] when
the drift wall is held ΔT above the far-field rock.

    planar_flux     exact: two flat faces into semi-infinite rock (2-D slab grid)
    radial_flux     fine-grid finite volume, cylinder in infinite rock (radial grid)
    section_flux    fine-grid finite volume, any drift shape in a 2-D section (3-D grid)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

DAY = 86400.0


def planar_flux(k, rho_c, dT, width_m, t_days):
    """q' = 2 W k ΔT / sqrt(pi alpha t): conduction from the top and bottom faces."""
    alpha = k / rho_c
    return 2.0 * width_m * k * dT / np.sqrt(np.pi * alpha * np.asarray(t_days) * DAY)


def radial_flux(k, rho_c, dT, radius_m, t_days, n=600, outer=200.0, dt_h=6.0):
    """Heat flow per metre from a round drift into rock, from a fine 1-D radial model.

    The rock is split into many thin rings that grow outward, and the drift wall is held dT
    above the rock. Each implicit time step is one solve with a pre-factorised matrix. The heat
    flow is read off the conductance between the wall and the first ring at each requested day.
    """
    e = np.geomspace(radius_m, outer, n + 1)
    rc = 0.5 * (e[:-1] + e[1:])
    C = rho_c * np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    G = 2 * np.pi * k * e[1:-1] / (rc[1:] - rc[:-1])
    Gw = 2 * np.pi * k * radius_m / (rc[0] - radius_m)
    dt = dt_h * 3600.0
    main = C / dt + np.r_[G, 0] + np.r_[0, G]
    main[0] += Gw
    lu = splu(sp.diags([main, -G, -G], [0, 1, -1], format="csc"))
    T, out, want = np.zeros(n), {}, sorted(np.atleast_1d(t_days))
    for s in range(1, int(round(max(want) * DAY / dt)) + 1):
        b = C / dt * T
        b[0] += Gw * dT
        T = lu.solve(b)
        for d in want:
            if abs(s * dt - d * DAY) < 1e-6:
                out[d] = Gw * (dT - T[0])
    return np.array([out[d] for d in np.atleast_1d(t_days)])


def section_flux(k, rho_c, dT, width_m, height_m, t_days, h=0.25, half_extent=60.0):
    """Heat flow per metre from a rectangular drift into rock, from a fine 2-D cross-section model.

    A uniform square grid surrounds the drift opening, whose wall is held dT above the rock. The
    outer edge stays at the rock temperature. Cells touching the wall are only half a cell from
    it, so their link to the wall is twice as strong. The heat flow is summed over every wall
    link after each daily step.
    """
    n = int(2 * half_extent / h)
    x = (np.arange(n) + 0.5) * h - half_extent
    X, Y = np.meshgrid(x, x)
    wall = (np.abs(X) <= width_m / 2) & (np.abs(Y) <= height_m / 2)
    act = ~wall
    act[0, :] = act[-1, :] = act[:, 0] = act[:, -1] = False       # far boundary at 0
    idx = -np.ones((n, n), int)
    idx[act] = np.arange(act.sum())
    N, dt = act.sum(), DAY
    cap = np.full(N, rho_c * h * h / dt)
    diag, rows, cols, vals, b0, wall_terms = cap.copy(), [], [], [], np.zeros(N), []
    ii, jj = np.nonzero(act)
    me = idx[ii, jj]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        w = wall[ni, nj]
        g = np.where(w, 2 * k, k)                      # half-cell distance to the wall face
        np.add.at(diag, me, g)
        nb = act[ni, nj]
        rows += list(me[nb]); cols += list(idx[ni, nj][nb]); vals += list(-g[nb])
        np.add.at(b0, me[w], g[w] * dT)
        wall_terms.append((me[w], g[w]))
    A = sp.csc_matrix((np.r_[diag, vals], (np.r_[np.arange(N), rows], np.r_[np.arange(N), cols])),
                      shape=(N, N))
    lu, T, out = splu(A), np.zeros(N), {}
    want = [int(d) for d in np.atleast_1d(t_days)]
    for d in range(1, max(want) + 1):
        T = lu.solve(cap * T + b0)
        if d in want:
            out[d] = sum((g * (dT - T[m])).sum() for m, g in wall_terms)
    return np.array([out[d] for d in want])

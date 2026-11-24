"""Multi-view feature triangulation: linear init + Gauss-Newton on inverse depth.

The feature is parameterised in the *anchor* camera (first observation) as
``(alpha, beta, rho) = (x/z, y/z, 1/z)``.  For observing camera ``i`` with
``p_ci = R_i p_a + t_i`` (``R_i = R_wi^T R_wa``, ``t_i = R_wi^T (p_wa - p_wi)``)::

    h_i = R_i [alpha, beta, 1]^T + rho t_i          (= rho * p_ci)
    z_i = h_i[:2] / h_i[2]

which is well-conditioned even for distant points (rho -> 0), as in
Mourikis & Roumeliotis (2007), Appendix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangulationResult:
    ok: bool
    p_w: np.ndarray            # world point
    anchor_inv_depth: np.ndarray  # (alpha, beta, rho)
    rms_error: float           # normalized-coordinate RMS reprojection error
    iterations: int


def triangulate_linear(R_wc: list, p_wc: list, obs: np.ndarray) -> np.ndarray:
    """Least-squares intersection of bearing rays (midpoint method) in world frame."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for R, p, z in zip(R_wc, p_wc, obs):
        d = R @ np.array([z[0], z[1], 1.0])
        d /= np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    return np.linalg.solve(A + 1e-12 * np.eye(3), b)


def triangulate_gn(R_wc: list, p_wc: list, obs: np.ndarray, max_iters: int = 15,
                   min_depth: float = 0.1, max_depth: float = 60.0,
                   max_rms: float = 0.02) -> TriangulationResult:
    """Triangulate one feature from >=2 camera poses (world<-camera) and normalized obs.

    ``max_rms`` is in normalized image units (0.02 ~ 8 px at f=400).
    """
    obs = np.asarray(obs, dtype=float)
    n = len(obs)
    fail = TriangulationResult(False, np.zeros(3), np.zeros(3), np.inf, 0)
    if n < 2:
        return fail
    R_wa, p_wa = R_wc[0], p_wc[0]
    Rs = [R.T @ R_wa for R in R_wc]
    ts = [R.T @ (p_wa - p) for R, p in zip(R_wc, p_wc)]

    # --- linear initialisation, check ray geometry
    p_lin = triangulate_linear(R_wc, p_wc, obs)
    p_a = R_wa.T @ (p_lin - p_wa)
    if p_a[2] < min_depth:
        # fall back to a nominal depth along the anchor ray
        p_a = np.array([obs[0, 0], obs[0, 1], 1.0]) * 3.0
    x = np.array([p_a[0] / p_a[2], p_a[1] / p_a[2], 1.0 / p_a[2]])

    def residuals(x):
        r = np.zeros(2 * n)
        J = np.zeros((2 * n, 3))
        for i in range(n):
            h = Rs[i] @ np.array([x[0], x[1], 1.0]) + x[2] * ts[i]
            hz = h[2] if abs(h[2]) > 1e-9 else 1e-9
            r[2 * i:2 * i + 2] = obs[i] - h[:2] / hz
            dz_dh = np.array([[1 / hz, 0, -h[0] / hz ** 2], [0, 1 / hz, -h[1] / hz ** 2]])
            dh_dx = np.column_stack([Rs[i][:, 0], Rs[i][:, 1], ts[i]])
            J[2 * i:2 * i + 2] = dz_dh @ dh_dx   # d(prediction)/dx
        return r, J

    lam = 1e-3
    r, J = residuals(x)
    cost = r @ r
    it = 0
    for it in range(1, max_iters + 1):
        H = J.T @ J
        g = J.T @ r
        dx = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        x_new = x + dx
        r_new, J_new = residuals(x_new)
        cost_new = r_new @ r_new
        if cost_new < cost:
            x, r, J, cost = x_new, r_new, J_new, cost_new
            lam = max(lam * 0.1, 1e-10)
            if np.linalg.norm(dx) < 1e-9:
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                break

    if x[2] <= 0:
        return fail
    p_a = np.array([x[0], x[1], 1.0]) / x[2]
    p_w = R_wa @ p_a + p_wa
    for R, p in zip(R_wc, p_wc):
        d = (R.T @ (p_w - p))[2]
        if d < min_depth or d > max_depth:
            return TriangulationResult(False, p_w, x, np.sqrt(cost / n), it)
    rms = float(np.sqrt(cost / n))
    return TriangulationResult(rms < max_rms, p_w, x, rms, it)

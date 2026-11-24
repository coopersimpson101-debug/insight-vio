"""Pyramidal Lucas-Kanade (KLT) optical flow, hand-written and vectorised over points.

For each point we seek the displacement ``d`` minimising
``sum_{x in W} (J(x + d) - I(x))^2``.  Linearising around the current estimate
and approximating the gradient of ``J`` by the (fixed) template gradient of ``I``
(Bouguet 2000; inverse-compositional flavour) gives the Gauss-Newton step::

    G = sum [gx^2, gx gy; gx gy, gy^2]       (computed once per level)
    b = sum [gx, gy]^T (I(x) - J(x + d))
    d <- d + G^{-1} b

run coarse-to-fine over a Gaussian pyramid (``d`` doubles between levels).
Pixel convention: integer coordinates are pixel centres; level ``L`` pixel ``i``
corresponds to level-0 pixel ``2^L i``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_K5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


def blur5(img: np.ndarray) -> np.ndarray:
    """Separable [1 4 6 4 1]/16 Gaussian blur with reflect padding."""
    p = np.pad(img, ((0, 0), (2, 2)), mode="reflect")
    h = sum(_K5[i] * p[:, i:i + img.shape[1]] for i in range(5))
    p = np.pad(h, ((2, 2), (0, 0)), mode="reflect")
    return sum(_K5[i] * p[i:i + img.shape[0], :] for i in range(5)).astype(np.float32)


def central_gradients(img: np.ndarray):
    p = np.pad(img, 1, mode="edge")
    gx = 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
    gy = 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    return gx.astype(np.float32), gy.astype(np.float32)


@dataclass
class Pyramid:
    levels: list          # images, level 0 = full resolution
    gx: list
    gy: list


def build_pyramid(img: np.ndarray, n_levels: int = 4) -> Pyramid:
    lv = [np.asarray(img, dtype=np.float32)]
    for _ in range(1, n_levels):
        prev = lv[-1]
        if min(prev.shape) < 16:
            break
        lv.append(blur5(prev)[::2, ::2].copy())
    gxs, gys = zip(*(central_gradients(l) for l in lv))
    return Pyramid(lv, list(gxs), list(gys))


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at float coordinates (clamped to the image)."""
    H, W = img.shape
    x = np.clip(x, 0.0, W - 1.001)
    y = np.clip(y, 0.0, H - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    ax = x - x0
    ay = y - y0
    flat = img.ravel()
    i00 = y0 * W + x0
    v00 = flat[i00]
    v01 = flat[i00 + 1]
    v10 = flat[i00 + W]
    v11 = flat[i00 + W + 1]
    return (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay


def track(prev: Pyramid, nxt: Pyramid, pts: np.ndarray, guess: np.ndarray | None = None,
          win: int = 15, iters: int = 20, eps: float = 0.01, min_eig: float = 1e-3,
          max_residual: float = 15.0):
    """Track ``pts`` (N,2) from ``prev`` to ``nxt``.

    Returns ``(new_pts (N,2), status (N,) bool, residual (N,))`` where residual is the
    mean absolute intensity error over the window at level 0.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    N = len(pts)
    if N == 0:
        return pts.copy(), np.zeros(0, bool), np.zeros(0)
    n_lv = min(len(prev.levels), len(nxt.levels))
    half = win // 2
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1]
    ox = ox.ravel().astype(np.float64)[None, :]
    oy = oy.ravel().astype(np.float64)[None, :]
    top = n_lv - 1
    d = np.zeros((N, 2)) if guess is None else (np.asarray(guess, float) - pts)
    d = d / (2 ** top)
    status = np.ones(N, dtype=bool)
    for L in range(top, -1, -1):
        s = 2.0 ** L
        I, J = prev.levels[L], nxt.levels[L]
        p = pts / s
        X = p[:, 0:1] + ox
        Y = p[:, 1:2] + oy
        T = bilinear(I, X, Y)
        gx = bilinear(prev.gx[L], X, Y)
        gy = bilinear(prev.gy[L], X, Y)
        a = (gx * gx).sum(1)
        b = (gx * gy).sum(1)
        c = (gy * gy).sum(1)
        det = a * c - b * b
        lam_min = (0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b * b)) / ox.size
        ok_g = (lam_min > min_eig) & (det > 1e-9)
        det_s = np.where(ok_g, det, 1.0)
        active = ok_g.copy()
        for _ in range(iters):
            if not active.any():
                break
            Jw = bilinear(J, X + d[:, 0:1], Y + d[:, 1:2])
            e = T - Jw
            bx = (gx * e).sum(1)
            by = (gy * e).sum(1)
            ddx = (c * bx - b * by) / det_s
            ddy = (-b * bx + a * by) / det_s
            ddx = np.where(active, ddx, 0.0)
            ddy = np.where(active, ddy, 0.0)
            d[:, 0] += ddx
            d[:, 1] += ddy
            active &= (ddx * ddx + ddy * ddy) > eps * eps
        if L == 0:
            status &= ok_g
        if L > 0:
            d *= 2.0
    new = pts + d
    H, W = nxt.levels[0].shape
    status &= (new[:, 0] >= 0) & (new[:, 0] <= W - 1) & (new[:, 1] >= 0) & (new[:, 1] <= H - 1)
    status &= np.all(np.isfinite(new), axis=1)
    # final photometric residual at full resolution
    X = pts[:, 0:1] + ox
    Y = pts[:, 1:2] + oy
    res = np.abs(bilinear(prev.levels[0], X, Y) -
                 bilinear(nxt.levels[0], X + d[:, 0:1], Y + d[:, 1:2])).mean(1)
    status &= res < max_residual
    return new, status, res


def track_bidirectional(prev: Pyramid, nxt: Pyramid, pts: np.ndarray, guess=None,
                        fb_threshold: float = 0.5, **kw):
    """Forward-backward checked KLT: keep points that return within ``fb_threshold`` px."""
    fwd, st1, _ = track(prev, nxt, pts, guess, **kw)
    bwd, st2, _ = track(nxt, prev, fwd, pts, **kw)
    fb = np.linalg.norm(bwd - pts, axis=1)
    return fwd, st1 & st2 & (fb < fb_threshold), fb

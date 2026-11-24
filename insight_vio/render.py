"""Tiny ray-caster that renders the synthetic room as grayscale images.

Every pixel's ray is intersected with the inside of an axis-aligned box (the
room).  Walls carry a procedural texture (multi-octave value noise + random
"posters"), giving a corner-rich scene so the *real* image front-end (FAST +
KLT + stereo matching) can be tested end-to-end against ground truth.
No lighting, no motion blur, no rolling shutter - this is a front-end smoke
test, not a photorealistic simulator.
"""
from __future__ import annotations

import numpy as np

from .camera import Camera


class RoomRenderer:
    def __init__(self, half: float = 4.0, height: float = 3.0, seed: int = 0):
        self.lo = np.array([-half, -half, 0.0])
        self.hi = np.array([half, half, height])
        rng = np.random.default_rng(seed)
        self.tables = [rng.uniform(0, 1, (256, 256)).astype(np.float32) for _ in range(3)]

    def _value_noise(self, table, u, v, scale):
        x, y = u / scale, v / scale
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        i0, j0 = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255
        i1, j1 = (i0 + 1) & 255, (j0 + 1) & 255
        a, b = table[i0, j0], table[i1, j0]
        c, d = table[i0, j1], table[i1, j1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def texture(self, u, v, face):
        off = face * 37.0
        t = 0.45 * self._value_noise(self.tables[0], u + off, v, 0.35)
        t += 0.35 * self._value_noise(self.tables[1], u, v + off, 0.09)
        # piecewise-constant "posters": sharp corners for FAST
        cell = self.tables[2][(np.floor(u / 0.21).astype(np.int64) + face * 11) & 255,
                              np.floor(v / 0.17).astype(np.int64) & 255]
        t += 0.5 * np.where(cell > 0.6, cell, 0.0)
        return t

    def render(self, cam: Camera, R_wc: np.ndarray, p_wc: np.ndarray) -> np.ndarray:
        H, W = cam.height, cam.width
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        xn = cam.pixel_to_normalized(np.column_stack([uu.ravel(), vv.ravel()]))
        d = np.column_stack([xn, np.ones(len(xn))]) @ R_wc.T
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(d > 0, self.hi, self.lo)
            t_axes = (bound - p_wc) / d
        t_axes = np.where(np.isfinite(t_axes) & (t_axes > 0), t_axes, np.inf)
        axis = np.argmin(t_axes, axis=1)
        t = t_axes[np.arange(len(axis)), axis]
        P = p_wc + d * t[:, None]
        # wall-plane coordinates
        u = np.where(axis == 0, P[:, 1], P[:, 0])
        v = np.where(axis == 2, P[:, 1], P[:, 2])
        face = axis * 2 + (d[np.arange(len(axis)), axis] > 0)
        img = self.texture(u, v, face)
        img = np.clip(img * 200.0 + 20.0, 0, 255)
        return img.reshape(H, W).astype(np.uint8)

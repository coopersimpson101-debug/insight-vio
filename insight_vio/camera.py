"""Pinhole camera models with radial-tangential or equidistant (fisheye) distortion.

All projection / undistortion is hand-written and vectorised over points so the
filter can work in *normalized* image coordinates ``(x/z, y/z)``.  Distortion
models and parameter names match Kalibr's ``pinhole-radtan`` and
``pinhole-equi`` so calibrations can be pasted straight in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"           # "none" | "radtan" | "equidistant"
    distortion: list = field(default_factory=list)
    R_bc: np.ndarray = field(default_factory=lambda: np.eye(3))  # camera -> body (IMU)
    p_bc: np.ndarray = field(default_factory=lambda: np.zeros(3))  # camera origin in body

    # ------------------------------------------------------------ distortion
    def distort(self, xn: np.ndarray) -> np.ndarray:
        """Normalized undistorted coordinates (N,2) -> normalized distorted (N,2)."""
        xn = np.atleast_2d(np.asarray(xn, dtype=float))
        x, y = xn[:, 0], xn[:, 1]
        d = list(self.distortion) + [0.0] * 4
        if self.distortion_model in ("none", "", None):
            return xn.copy()
        if self.distortion_model == "radtan":
            k1, k2, p1, p2 = d[:4]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            return np.stack([xd, yd], axis=1)
        if self.distortion_model == "equidistant":
            k1, k2, k3, k4 = d[:4]
            r = np.sqrt(x * x + y * y)
            th = np.arctan(r)
            th2 = th * th
            thd = th * (1 + k1 * th2 + k2 * th2 ** 2 + k3 * th2 ** 3 + k4 * th2 ** 4)
            scale = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
            return np.stack([x * scale, y * scale], axis=1)
        raise ValueError(f"unknown distortion model {self.distortion_model!r}")

    def undistort(self, xd: np.ndarray, iters: int = 20) -> np.ndarray:
        """Invert :meth:`distort` with vectorised Gauss-Newton (numeric 2x2 Jacobian)."""
        xd = np.atleast_2d(np.asarray(xd, dtype=float))
        if self.distortion_model in ("none", "", None):
            return xd.copy()
        x = xd.copy()
        h = 1e-7
        for _ in range(iters):
            f = self.distort(x) - xd
            if np.max(np.abs(f)) < 1e-12:
                break
            fx = (self.distort(x + [h, 0]) - xd - f) / h
            fy = (self.distort(x + [0, h]) - xd - f) / h
            a, c = fx[:, 0], fx[:, 1]
            b, d = fy[:, 0], fy[:, 1]
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (d * f[:, 0] - b * f[:, 1]) / det
            dy = (-c * f[:, 0] + a * f[:, 1]) / det
            x = x - np.stack([dx, dy], axis=1)
        return x

    # ------------------------------------------------------------ pixel <-> normalized
    def pixel_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        xd = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], axis=1)
        return self.undistort(xd)

    def normalized_to_pixel(self, xn: np.ndarray) -> np.ndarray:
        xd = self.distort(xn)
        return np.stack([self.fx * xd[:, 0] + self.cx, self.fy * xd[:, 1] + self.cy], axis=1)

    def project(self, p_c: np.ndarray):
        """Project camera-frame points (N,3). Returns (pixels (N,2), valid mask)."""
        p_c = np.atleast_2d(np.asarray(p_c, dtype=float))
        z = p_c[:, 2]
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        xn = p_c[:, :2] / zs[:, None]
        uv = self.normalized_to_pixel(xn)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= self.width - 1) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] <= self.height - 1)
        return uv, valid

    @property
    def mean_focal(self) -> float:
        return 0.5 * (self.fx + self.fy)

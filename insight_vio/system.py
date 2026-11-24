"""End-to-end VIO system: IMU buffer + stereo front-end + MSCKF.

Used by ``scripts/run_euroc.py`` and ``scripts/run_live.py``.
"""
from __future__ import annotations

import numpy as np

from .config import VioConfig
from .frontend import StereoFrontend
from .imu import Preintegrator, iterate_imu_segments
from .msckf import Msckf


class VioSystem:
    def __init__(self, cfg: VioConfig, stereo: bool = True, static_init_seconds: float = 1.0):
        self.cfg = cfg
        cams = cfg.cameras if stereo else cfg.cameras[:1]
        self.stereo = stereo and len(cams) > 1
        self.frontend = StereoFrontend(cams, cfg.frontend)
        fcfg = cfg.filter
        if not self.stereo:
            fcfg.cameras = fcfg.cameras[:1]
        self.filter = Msckf(fcfg)
        self.static_init_seconds = static_init_seconds
        self.imu_t, self.imu_w, self.imu_a = [], [], []
        self.last_cam_t = None

    def add_imu(self, t: float, w, a) -> None:
        self.imu_t.append(t)
        self.imu_w.append(np.asarray(w, float))
        self.imu_a.append(np.asarray(a, float))
        if self.filter.initialized:
            self.filter.add_imu(t, w, a)

    def initialize_from_state(self, t, R, p, v, bg=None, ba=None, P0=None) -> None:
        self.filter.initialize(t, R, p, v, bg, ba, P0)
        for ti, w, a in zip(self.imu_t, self.imu_w, self.imu_a):
            if ti >= t - 0.05:
                self.filter.add_imu(ti, w, a)

    def _try_static_init(self, t: float) -> bool:
        ts = np.asarray(self.imu_t)
        if len(ts) < 20 or ts[-1] - ts[0] < self.static_init_seconds:
            return False
        sel = ts >= ts[-1] - self.static_init_seconds
        acc = np.asarray(self.imu_a)[sel]
        if np.std(np.linalg.norm(acc, axis=1)) > 0.3:     # not static enough yet
            return False
        self.filter.initialize_static(t + self.cfg.time_offset, acc,
                                      np.asarray(self.imu_w)[sel])
        for ti, w, a in zip(self.imu_t, self.imu_w, self.imu_a):
            if ti >= t + self.cfg.time_offset - 0.05:
                self.filter.add_imu(ti, w, a)
        return True

    def _gyro_rotation(self, t0: float, t1: float):
        """Gyro-only rotation of the left camera between two frames (R_c1_c0)."""
        if t0 is None:
            return None
        pre = Preintegrator(self.filter.bg, np.zeros(3), self.cfg.noise)
        times = np.asarray(self.imu_t)
        for k, dt in iterate_imu_segments(times, t0 + self.cfg.time_offset,
                                          t1 + self.cfg.time_offset):
            pre.integrate(self.imu_w[k], self.imu_a[k], dt)
        R_bc = self.cfg.cameras[0].R_bc
        return R_bc.T @ pre.dR.T @ R_bc

    def process_images(self, t_cam: float, left, right=None):
        """Returns the current (t, R_wb, p_wb, v_wb) or None while initialising."""
        R_pred = self._gyro_rotation(self.last_cam_t, t_cam)
        meas = self.frontend.process(left, right if self.stereo else None, R_pred)
        self.last_cam_t = t_cam
        if not self.filter.initialized and not self._try_static_init(t_cam):
            return None
        self.filter.process_frame(t_cam, meas)
        # keep the raw IMU buffer bounded
        if len(self.imu_t) > 4000:
            del self.imu_t[:2000], self.imu_w[:2000], self.imu_a[:2000]
        return self.filter.pose()

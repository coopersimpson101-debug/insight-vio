"""Camera-IMU time synchronisation helpers.

Two separate problems:

1. **Clock mapping** - a sensor-internal counter (e.g. BMI088 sensortime, camera
   frame counter) vs. the host clock.  :class:`LinearClockModel` fits
   ``t_host = offset + (1 + drift) * t_sensor`` with exponentially-forgetting
   recursive least squares, which removes jitter from host-side timestamps taken
   in user space.
2. **Residual camera-IMU offset** ``td`` (``t_imu = t_cam + td``) caused by exposure
   mid-point, readout and driver latency.  Kalibr estimates it offline; the
   function :func:`estimate_time_offset` gives a quick independent check by
   cross-correlating gyro angular speed with the rotation speed seen by the camera.
"""
from __future__ import annotations

import numpy as np


class LinearClockModel:
    def __init__(self, forgetting: float = 0.999):
        self.lam = forgetting
        self.P = np.eye(2) * 1e6
        self.theta = np.zeros(2)       # [offset, slope]
        self.t0 = None
        self.n = 0

    def update(self, t_sensor: float, t_host: float) -> None:
        if self.t0 is None:
            self.t0 = (t_sensor, t_host)
        x = np.array([1.0, t_sensor - self.t0[0]])
        y = t_host - self.t0[1]
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.theta = self.theta + k * (y - x @ self.theta)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.n += 1

    def to_host(self, t_sensor: float) -> float:
        return self.t0[1] + self.theta[0] + self.theta[1] * (t_sensor - self.t0[0])

    @property
    def drift_ppm(self) -> float:
        return (self.theta[1] - 1.0) * 1e6


def estimate_time_offset(t_imu: np.ndarray, gyro_speed: np.ndarray, t_cam: np.ndarray,
                         cam_speed: np.ndarray, max_offset: float = 0.05,
                         step: float = 0.0005) -> float:
    """Return ``td`` maximising correlation of ``cam_speed(t)`` with ``gyro_speed(t + td)``.

    ``gyro_speed`` / ``cam_speed`` are angular-speed magnitudes (rad/s); rotation speed
    is frame-independent, so the camera-IMU extrinsic is not needed.
    """
    cam = (cam_speed - cam_speed.mean()) / (cam_speed.std() + 1e-12)
    offsets = np.arange(-max_offset, max_offset + 0.5 * step, step)
    scores = []
    for td in offsets:
        g = np.interp(t_cam + td, t_imu, gyro_speed)
        g = (g - g.mean()) / (g.std() + 1e-12)
        scores.append(float(np.mean(g * cam)))
    scores = np.array(scores)
    k = int(np.argmax(scores))
    # parabolic refinement
    if 0 < k < len(scores) - 1:
        a, b, c = scores[k - 1], scores[k], scores[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return float(offsets[k] + 0.5 * step * (a - c) / den)
    return float(offsets[k])

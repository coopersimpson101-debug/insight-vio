"""IMU noise model and on-manifold preintegration (Forster et al., T-RO 2017).

Preintegrated measurements between times i and j (biases held at ``b_hat``)::

    dR_ij = prod_k Exp((w_k - bg) dt)
    dv_ij = sum_k dR_ik (a_k - ba) dt
    dp_ij = sum_k dv_ik dt + 1/2 dR_ik (a_k - ba) dt^2

and the state prediction (world gravity ``g``, e.g. ``[0, 0, -9.81]``)::

    R_j = R_i dR_ij
    v_j = v_i + g dt_ij + R_i dv_ij
    p_j = p_i + v_i dt_ij + 1/2 g dt_ij^2 + R_i dp_ij

The 9x9 covariance (order ``[dphi, dv, dp]``) and the first-order bias
Jacobians ``d(dR,dv,dp)/d(bg,ba)`` are propagated recursively, exactly as in
Forster's appendix.  Noise parameters are *continuous-time densities*
(same units as Kalibr / EuRoC ``sensor.yaml``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import skew, so3_exp, so3_right_jacobian

GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class ImuNoise:
    gyro_noise: float = 1.6968e-4     # rad/s/sqrt(Hz)      (EuRoC ADIS16448 values)
    accel_noise: float = 2.0e-3       # m/s^2/sqrt(Hz)
    gyro_walk: float = 1.9393e-5      # rad/s^2/sqrt(Hz)
    accel_walk: float = 3.0e-3        # m/s^3/sqrt(Hz)


class Preintegrator:
    """Accumulates IMU samples into a relative-motion measurement with covariance."""

    def __init__(self, bg: np.ndarray, ba: np.ndarray, noise: ImuNoise):
        self.bg = np.array(bg, dtype=float)
        self.ba = np.array(ba, dtype=float)
        self.noise = noise
        self.dR = np.eye(3)
        self.dv = np.zeros(3)
        self.dp = np.zeros(3)
        self.dt = 0.0
        self.cov = np.zeros((9, 9))
        self.J_Rg = np.zeros((3, 3))
        self.J_vg = np.zeros((3, 3))
        self.J_va = np.zeros((3, 3))
        self.J_pg = np.zeros((3, 3))
        self.J_pa = np.zeros((3, 3))

    def integrate(self, w_m: np.ndarray, a_m: np.ndarray, dt: float) -> None:
        """Add one IMU sample held constant over ``dt`` seconds (zero-order hold)."""
        if dt <= 0.0:
            return
        w = np.asarray(w_m, dtype=float) - self.bg
        a = np.asarray(a_m, dtype=float) - self.ba
        dR = self.dR
        a_sk = skew(a)
        dRk = so3_exp(w * dt)
        Jr = so3_right_jacobian(w * dt)
        dt2 = dt * dt

        # --- covariance: eta_{k+1} = A eta_k + B n_k
        A = np.eye(9)
        A[0:3, 0:3] = dRk.T
        A[3:6, 0:3] = -dR @ a_sk * dt
        A[6:9, 0:3] = -0.5 * dR @ a_sk * dt2
        A[6:9, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 6))
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = dR * dt
        B[6:9, 3:6] = 0.5 * dR * dt2
        qg = self.noise.gyro_noise ** 2 / dt      # discrete-time variance
        qa = self.noise.accel_noise ** 2 / dt
        Q = np.diag([qg, qg, qg, qa, qa, qa])
        self.cov = A @ self.cov @ A.T + B @ Q @ B.T

        # --- bias Jacobians (use values from step k on the right-hand side)
        self.J_pa = self.J_pa + self.J_va * dt - 0.5 * dR * dt2
        self.J_pg = self.J_pg + self.J_vg * dt - 0.5 * dR @ a_sk @ self.J_Rg * dt2
        self.J_va = self.J_va - dR * dt
        self.J_vg = self.J_vg - dR @ a_sk @ self.J_Rg * dt
        self.J_Rg = dRk.T @ self.J_Rg - Jr * dt

        # --- mean
        Ra = dR @ a
        self.dp = self.dp + self.dv * dt + 0.5 * Ra * dt2
        self.dv = self.dv + Ra * dt
        self.dR = dR @ dRk
        self.dt += dt

    def corrected(self, bg_new: np.ndarray, ba_new: np.ndarray):
        """First-order bias correction without re-integration (Forster eq. 44)."""
        dbg = np.asarray(bg_new) - self.bg
        dba = np.asarray(ba_new) - self.ba
        dR = self.dR @ so3_exp(self.J_Rg @ dbg)
        dv = self.dv + self.J_vg @ dbg + self.J_va @ dba
        dp = self.dp + self.J_pg @ dbg + self.J_pa @ dba
        return dR, dv, dp

    def predict(self, R_i, p_i, v_i, g: np.ndarray = GRAVITY):
        """Predict the state at time j from the state at i."""
        T = self.dt
        R_j = R_i @ self.dR
        v_j = v_i + g * T + R_i @ self.dv
        p_j = p_i + v_i * T + 0.5 * g * T * T + R_i @ self.dp
        return R_j, p_j, v_j


def iterate_imu_segments(times: np.ndarray, t0: float, t1: float):
    """Yield ``(index, dt)`` for zero-order-hold IMU samples covering ``[t0, t1)``.

    Sample ``k`` is assumed valid on ``[times[k], times[k+1])``.  The last sample
    before ``t1`` is extrapolated if no later sample exists yet.
    """
    if t1 <= t0:
        return
    n = len(times)
    k = int(np.searchsorted(times, t0, side="right")) - 1
    k = max(k, 0)
    while k < n:
        seg_start = max(times[k], t0)
        seg_end = times[k + 1] if k + 1 < n else t1
        seg_end = min(seg_end, t1)
        if seg_end > seg_start:
            yield k, seg_end - seg_start
        if seg_end >= t1:
            break
        k += 1

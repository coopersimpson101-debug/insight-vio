"""Multi-State Constraint Kalman Filter (Mourikis & Roumeliotis, ICRA 2007).

State (nominal)::

    x = [ R_wb, p_wb, v_wb, b_g, b_a | (R_wb, p_wb)_clone_1 ... (R_wb, p_wb)_clone_N ]

Error state (dimension 15 + 6N), right/local perturbation on rotations::

    R = R_hat Exp(dtheta),  p = p_hat + dp,  v = v_hat + dv,  b = b_hat + db
    dx = [dtheta, dp, dv, dbg, dba | dtheta_1, dp_1, ... ]

* **Propagation** uses Forster preintegration between two camera frames (see
  :mod:`insight_vio.imu`) and its analytic error-state Jacobian ``F`` and
  noise ``Q`` (derivation in ``docs/msckf.md``).
* **Augmentation**: every camera frame clones the current IMU pose.
* **Update**: when a feature track ends (or its oldest clone is about to be
  marginalised) the feature is triangulated (Gauss-Newton inverse depth), the
  stacked reprojection residual ``r = H_x dx + H_f dp_f + n`` is projected onto
  the left null-space of ``H_f`` to remove the feature, chi-squared gated,
  QR-compressed, and applied in a standard EKF (Joseph form) update.
* Stereo: each observation carries a camera index; each camera has its own
  extrinsic ``(R_bc, p_bc)``.  Camera-IMU time offset ``t_imu = t_cam + td`` is a
  configurable parameter.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .imu import GRAVITY, ImuNoise, Preintegrator, iterate_imu_segments
from .lie import orthonormalize, skew, so3_exp
from .stats import chi2_quantile
from .triangulation import triangulate_gn

IMU_DIM = 15
CLONE_DIM = 6
I_TH, I_P, I_V, I_BG, I_BA = 0, 3, 6, 9, 12


def propagation_jacobians(pre: Preintegrator, R_i: np.ndarray, noise: ImuNoise):
    """Error-state transition ``F`` (15x15) and noise ``Q`` for one preintegrated interval.

    With right perturbations and ``dx = [dtheta, dp, dv, dbg, dba]``::

        dtheta_j = dR^T dtheta_i + J_Rg dbg
        dv_j     = dv_i - R_i [dv_ij]x dtheta_i + R_i (J_vg dbg + J_va dba)
        dp_j     = dp_i + T dv_i - R_i [dp_ij]x dtheta_i + R_i (J_pg dbg + J_pa dba)
        db_j     = db_i + random walk
    """
    T = pre.dt
    F = np.eye(IMU_DIM)
    F[I_TH:I_TH + 3, I_TH:I_TH + 3] = pre.dR.T
    F[I_TH:I_TH + 3, I_BG:I_BG + 3] = pre.J_Rg
    F[I_P:I_P + 3, I_TH:I_TH + 3] = -R_i @ skew(pre.dp)
    F[I_P:I_P + 3, I_V:I_V + 3] = np.eye(3) * T
    F[I_P:I_P + 3, I_BG:I_BG + 3] = R_i @ pre.J_pg
    F[I_P:I_P + 3, I_BA:I_BA + 3] = R_i @ pre.J_pa
    F[I_V:I_V + 3, I_TH:I_TH + 3] = -R_i @ skew(pre.dv)
    F[I_V:I_V + 3, I_BG:I_BG + 3] = R_i @ pre.J_vg
    F[I_V:I_V + 3, I_BA:I_BA + 3] = R_i @ pre.J_va
    # preintegration noise [phi, v, p] mapped into [theta, p, v]
    G = np.zeros((IMU_DIM, 9))
    G[I_TH:I_TH + 3, 0:3] = np.eye(3)
    G[I_V:I_V + 3, 3:6] = R_i
    G[I_P:I_P + 3, 6:9] = R_i
    Q = G @ pre.cov @ G.T
    Q[I_BG:I_BG + 3, I_BG:I_BG + 3] += np.eye(3) * noise.gyro_walk ** 2 * T
    Q[I_BA:I_BA + 3, I_BA:I_BA + 3] += np.eye(3) * noise.accel_walk ** 2 * T
    return F, Q


@dataclass
class CameraExtrinsics:
    R_bc: np.ndarray
    p_bc: np.ndarray


@dataclass
class MsckfConfig:
    cameras: list = field(default_factory=lambda: [CameraExtrinsics(np.eye(3), np.zeros(3))])
    noise: ImuNoise = field(default_factory=ImuNoise)
    gravity: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    max_clones: int = 11
    min_track_clones: int = 2          # distinct clones needed to use a feature
    obs_sigma: float = 1.0 / 400.0     # measurement std in normalized image units
    chi2_prob: float = 0.95
    chi2_multiplier: float = 1.0
    time_offset: float = 0.0           # t_imu = t_cam + time_offset
    max_triang_rms: float = 0.02
    min_depth: float = 0.15
    max_depth: float = 60.0
    max_features_per_update: int = 400


@dataclass
class FeatureTrack:
    obs: "OrderedDict[int, list]" = field(default_factory=OrderedDict)  # clone_id -> [(cam, xy)]


class Msckf:
    def __init__(self, config: MsckfConfig):
        self.cfg = config
        self.initialized = False
        self.t = 0.0
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P = np.eye(IMU_DIM) * 1e-4
        self.clones: "OrderedDict[int, tuple]" = OrderedDict()  # id -> (t, R, p)
        self.features: dict[int, FeatureTrack] = {}
        self._imu_t: list[float] = []
        self._imu_w: list[np.ndarray] = []
        self._imu_a: list[np.ndarray] = []
        self._next_clone_id = 0
        self.stats = {"updates": 0, "features_used": 0, "features_rejected_chi2": 0,
                      "features_failed_triang": 0}

    # ------------------------------------------------------------------ setup
    def initialize(self, t: float, R: np.ndarray, p: np.ndarray, v: np.ndarray,
                   bg=None, ba=None, P0: np.ndarray | None = None) -> None:
        self.t = float(t)
        self.R, self.p, self.v = np.array(R, float), np.array(p, float), np.array(v, float)
        self.bg = np.zeros(3) if bg is None else np.array(bg, float)
        self.ba = np.zeros(3) if ba is None else np.array(ba, float)
        if P0 is None:
            sig = np.r_[[1e-3] * 3, [1e-3] * 3, [0.05] * 3, [0.01] * 3, [0.05] * 3]
            P0 = np.diag(sig ** 2)
        self.P = np.array(P0, float)
        self.clones.clear()
        self.features.clear()
        self.initialized = True

    def initialize_static(self, t: float, accel_samples: np.ndarray, gyro_samples: np.ndarray,
                          P0: np.ndarray | None = None) -> None:
        """Gravity-aligned init from a stationary IMU window (yaw is unobservable -> 0)."""
        a_mean = np.mean(accel_samples, axis=0)
        z_b = a_mean / np.linalg.norm(a_mean)         # gravity "up" in body frame
        # rotation taking body z_b to world +z, with zero yaw
        x_b = np.array([1.0, 0, 0]) - z_b[0] * z_b
        if np.linalg.norm(x_b) < 1e-3:
            x_b = np.array([0, 1.0, 0]) - z_b[1] * z_b
        x_b /= np.linalg.norm(x_b)
        y_b = np.cross(z_b, x_b)
        R_wb = np.vstack([x_b, y_b, z_b])             # rows: world axes in body coords
        bg = np.mean(gyro_samples, axis=0)
        self.initialize(t, R_wb, np.zeros(3), np.zeros(3), bg=bg, ba=np.zeros(3), P0=P0)

    # ------------------------------------------------------------------ IMU
    def add_imu(self, t: float, w: np.ndarray, a: np.ndarray) -> None:
        self._imu_t.append(float(t))
        self._imu_w.append(np.asarray(w, float))
        self._imu_a.append(np.asarray(a, float))

    def _trim_imu(self, t_keep: float) -> None:
        times = self._imu_t
        k = int(np.searchsorted(np.asarray(times), t_keep, side="right")) - 1
        if k > 0:
            del self._imu_t[:k], self._imu_w[:k], self._imu_a[:k]

    def propagate(self, t_new: float) -> None:
        if t_new <= self.t or not self._imu_t:
            return
        pre = Preintegrator(self.bg, self.ba, self.cfg.noise)
        times = np.asarray(self._imu_t)
        for k, dt in iterate_imu_segments(times, self.t, t_new):
            pre.integrate(self._imu_w[k], self._imu_a[k], dt)
        if pre.dt <= 0:
            return
        F, Q = propagation_jacobians(pre, self.R, self.cfg.noise)
        self.R, self.p, self.v = pre.predict(self.R, self.p, self.v, self.cfg.gravity)
        self.R = orthonormalize(self.R)

        P = self.P
        Pii = P[:IMU_DIM, :IMU_DIM]
        P[:IMU_DIM, :IMU_DIM] = F @ Pii @ F.T + Q
        if P.shape[0] > IMU_DIM:
            P[:IMU_DIM, IMU_DIM:] = F @ P[:IMU_DIM, IMU_DIM:]
            P[IMU_DIM:, :IMU_DIM] = P[:IMU_DIM, IMU_DIM:].T
        self.P = 0.5 * (P + P.T)
        self.t = t_new
        self._trim_imu(t_new)

    # ------------------------------------------------------------------ clones
    def _augment(self) -> int:
        cid = self._next_clone_id
        self._next_clone_id += 1
        self.clones[cid] = (self.t, self.R.copy(), self.p.copy())
        n = self.P.shape[0]
        J = np.zeros((CLONE_DIM, n))
        J[0:3, I_TH:I_TH + 3] = np.eye(3)
        J[3:6, I_P:I_P + 3] = np.eye(3)
        Pn = np.zeros((n + CLONE_DIM, n + CLONE_DIM))
        Pn[:n, :n] = self.P
        PJ = self.P @ J.T
        Pn[:n, n:] = PJ
        Pn[n:, :n] = PJ.T
        Pn[n:, n:] = J @ PJ
        self.P = Pn
        return cid

    def _clone_index(self, cid: int) -> int:
        return IMU_DIM + CLONE_DIM * list(self.clones.keys()).index(cid)

    def _remove_clone(self, cid: int) -> None:
        idx = self._clone_index(cid)
        keep = np.r_[0:idx, idx + CLONE_DIM:self.P.shape[0]]
        self.P = self.P[np.ix_(keep, keep)]
        del self.clones[cid]
        for f in self.features.values():
            f.obs.pop(cid, None)

    # ------------------------------------------------------------------ frames
    def process_frame(self, t_cam: float, measurements: dict) -> None:
        """Process one (stereo) frame.

        ``measurements``: ``{feature_id: [(cam_index, np.array([x_n, y_n])), ...]}`` in
        normalized (undistorted) image coordinates.
        """
        if not self.initialized:
            raise RuntimeError("filter not initialised")
        t = t_cam + self.cfg.time_offset
        self.propagate(t)
        cid = self._augment()

        for fid, obs in measurements.items():
            tr = self.features.get(fid)
            if tr is None:
                tr = self.features[fid] = FeatureTrack()
            tr.obs[cid] = [(int(c), np.asarray(z, float)) for c, z in obs]

        # features whose track ended this frame
        lost = [fid for fid, f in self.features.items() if cid not in f.obs]
        to_use = [fid for fid in lost if len(self.features[fid].obs) >= self.cfg.min_track_clones]
        for fid in lost:
            if fid not in to_use:
                del self.features[fid]

        # marginalise the oldest clone if the window is full
        marg_cid = None
        if len(self.clones) > self.cfg.max_clones:
            marg_cid = next(iter(self.clones))
            for fid, f in self.features.items():
                if marg_cid in f.obs and fid not in to_use:
                    if len(f.obs) >= self.cfg.min_track_clones:
                        to_use.append(fid)

        self._update(to_use[: self.cfg.max_features_per_update])
        for fid in to_use:
            self.features.pop(fid, None)
        if marg_cid is not None:
            self._remove_clone(marg_cid)

    # ------------------------------------------------------------------ update
    def _camera_pose(self, cid: int, cam: int):
        _, R_wb, p_wb = self.clones[cid]
        ex = self.cfg.cameras[cam]
        return R_wb @ ex.R_bc, p_wb + R_wb @ ex.p_bc

    def measurement_jacobian(self, fid: int):
        """Triangulate feature ``fid`` and return ``(r, H_x, H_f, p_f)`` or None."""
        f = self.features[fid]
        R_list, p_list, z_list, keys = [], [], [], []
        for cid, obs in f.obs.items():
            for cam, z in obs:
                R_wc, p_wc = self._camera_pose(cid, cam)
                R_list.append(R_wc)
                p_list.append(p_wc)
                z_list.append(z)
                keys.append((cid, cam))
        tri = triangulate_gn(R_list, p_list, np.array(z_list), min_depth=self.cfg.min_depth,
                             max_depth=self.cfg.max_depth, max_rms=self.cfg.max_triang_rms)
        if not tri.ok:
            self.stats["features_failed_triang"] += 1
            return None
        p_f = tri.p_w
        m = len(z_list)
        n = self.P.shape[0]
        H_x = np.zeros((2 * m, n))
        H_f = np.zeros((2 * m, 3))
        r = np.zeros(2 * m)
        for i, ((cid, cam), z) in enumerate(zip(keys, z_list)):
            _, R_wb, p_wb = self.clones[cid]
            ex = self.cfg.cameras[cam]
            p_b = R_wb.T @ (p_f - p_wb)
            p_c = ex.R_bc.T @ (p_b - ex.p_bc)
            Jp = np.array([[1 / p_c[2], 0, -p_c[0] / p_c[2] ** 2],
                           [0, 1 / p_c[2], -p_c[1] / p_c[2] ** 2]])
            JR = Jp @ ex.R_bc.T
            idx = self._clone_index(cid)
            H_x[2 * i:2 * i + 2, idx:idx + 3] = JR @ skew(p_b)
            H_x[2 * i:2 * i + 2, idx + 3:idx + 6] = -JR @ R_wb.T
            H_f[2 * i:2 * i + 2] = JR @ R_wb.T
            r[2 * i:2 * i + 2] = z - p_c[:2] / p_c[2]
        return r, H_x, H_f, p_f

    def _feature_jacobian(self, fid: int):
        """Return (r_o, H_o) after null-space projection, or None."""
        res = self.measurement_jacobian(fid)
        if res is None:
            return None
        r, H_x, H_f, _ = res
        # left null-space of H_f via full QR: H_f = Q [R1; 0]  ->  A = Q[:, 3:]
        Q, _ = np.linalg.qr(H_f, mode="complete")
        A = Q[:, 3:]
        return A.T @ r, A.T @ H_x

    def _update(self, fids: list) -> None:
        if not fids:
            return
        sigma2 = self.cfg.obs_sigma ** 2
        H_blocks, r_blocks = [], []
        for fid in fids:
            res = self._feature_jacobian(fid)
            if res is None:
                continue
            r_o, H_o = res
            if r_o.size == 0:
                continue
            # chi-squared gating with the innovation covariance of this feature
            S = H_o @ self.P @ H_o.T + sigma2 * np.eye(len(r_o))
            gamma = float(r_o @ np.linalg.solve(S, r_o))
            if gamma > self.cfg.chi2_multiplier * chi2_quantile(self.cfg.chi2_prob, len(r_o)):
                self.stats["features_rejected_chi2"] += 1
                continue
            H_blocks.append(H_o)
            r_blocks.append(r_o)
            self.stats["features_used"] += 1
        if not H_blocks:
            return
        H = np.vstack(H_blocks)
        r = np.concatenate(r_blocks)
        # measurement compression (QR) when there are more rows than states
        if H.shape[0] > H.shape[1]:
            Qh, Rh = np.linalg.qr(H, mode="reduced")
            H = Rh
            r = Qh.T @ r
        n = self.P.shape[0]
        Rn = sigma2 * np.eye(H.shape[0])
        S = H @ self.P @ H.T + Rn
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ r
        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rn @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._apply_correction(dx)
        self.stats["updates"] += 1

    def _apply_correction(self, dx: np.ndarray) -> None:
        self.R = orthonormalize(self.R @ so3_exp(dx[I_TH:I_TH + 3]))
        self.p = self.p + dx[I_P:I_P + 3]
        self.v = self.v + dx[I_V:I_V + 3]
        self.bg = self.bg + dx[I_BG:I_BG + 3]
        self.ba = self.ba + dx[I_BA:I_BA + 3]
        for i, cid in enumerate(list(self.clones.keys())):
            idx = IMU_DIM + CLONE_DIM * i
            t, R, p = self.clones[cid]
            self.clones[cid] = (t, orthonormalize(R @ so3_exp(dx[idx:idx + 3])),
                                p + dx[idx + 3:idx + 6])

    # ------------------------------------------------------------------ output
    def pose(self):
        """Current IMU pose (t, R_wb, p_wb, v_wb)."""
        return self.t, self.R.copy(), self.p.copy(), self.v.copy()

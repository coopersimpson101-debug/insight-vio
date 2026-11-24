"""Synthetic visual-inertial data for a head-mounted stereo rig.

* Smooth head-like 6DoF trajectory (sum of sinusoids: slow sway/steps in
  position, large yaw "looking around", smaller pitch/roll nods).
* IMU at ``imu_rate`` with white noise and bias random walk.  Gyro sample ``k``
  is the *average* angular rate over ``[t_k, t_k+1)`` and accel sample ``k`` is
  the average specific force that reproduces ``v_{k+1}`` exactly, so a
  zero-order-hold integrator reproduces ground truth up to O(dt^3) in position
  (this isolates estimator error from simulator discretisation error).
* Landmarks on the walls/floor/ceiling of a room plus a few interior pillars.
* Stereo pinhole measurements with Gaussian pixel noise and finite track length,
  returned in the same ``{feature_id: [(cam, xy_normalized)]}`` format the
  front-end produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .imu import GRAVITY, ImuNoise
from .lie import so3_log

# Body (IMU) frame: x forward, y left, z up.  Camera: z forward, x right, y down.
R_BC_FORWARD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), \
        np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class HeadTrajectory:
    """Analytic head-like motion. Position terms: (amplitude [m], omega [rad/s], phase)."""
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.6]))
    pos_terms: tuple = (
        ((0.60, 0.31, 0.0), (0.15, 1.30, 0.5)),     # x
        ((0.50, 0.23, 1.0), (0.10, 1.70, 0.2)),     # y
        ((0.05, 0.90, 0.3), (0.02, 3.10, 0.0)),     # z (bob)
    )
    yaw_terms: tuple = ((0.9, 0.35, 0.0), (0.3, 1.10, 0.7))
    pitch_terms: tuple = ((0.25, 0.60, 0.4), (0.08, 1.70, 0.0))
    roll_terms: tuple = ((0.12, 0.90, 1.2),)

    @staticmethod
    def _eval(terms, t, deriv=0):
        out = 0.0
        for a, w, ph in terms:
            if deriv == 0:
                out += a * np.sin(w * t + ph)
            elif deriv == 1:
                out += a * w * np.cos(w * t + ph)
            else:
                out += -a * w * w * np.sin(w * t + ph)
        return out

    def position(self, t):
        return self.center + np.array([self._eval(tt, t) for tt in self.pos_terms])

    def velocity(self, t):
        return np.array([self._eval(tt, t, 1) for tt in self.pos_terms])

    def acceleration(self, t):
        return np.array([self._eval(tt, t, 2) for tt in self.pos_terms])

    def rotation(self, t):
        return _rot_zyx(self._eval(self.yaw_terms, t), self._eval(self.pitch_terms, t),
                        self._eval(self.roll_terms, t))


@dataclass
class SimConfig:
    duration: float = 20.0
    imu_rate: float = 200.0
    cam_rate: float = 20.0
    seed: int = 0
    noise: ImuNoise = field(default_factory=ImuNoise)
    gyro_bias0_std: float = 0.01      # rad/s
    accel_bias0_std: float = 0.05     # m/s^2
    pixel_noise: float = 0.5          # px
    width: int = 640
    height: int = 480
    focal: float = 400.0
    baseline: float = 0.064           # m (roughly IPD)
    stereo: bool = True
    n_wall_landmarks: int = 1500
    n_pillar_landmarks: int = 300
    room_half: float = 4.0
    room_height: float = 3.0
    max_features: int = 120
    track_len_range: tuple = (4, 25)


@dataclass
class SimData:
    imu_t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    gt_t: np.ndarray           # ground truth at IMU rate
    gt_p: np.ndarray
    gt_R: np.ndarray
    gt_v: np.ndarray
    gt_bg: np.ndarray
    gt_ba: np.ndarray
    cam_t: np.ndarray
    cam_idx: np.ndarray        # index into gt arrays for each camera frame
    frames: list               # list of {fid: [(cam, xy_norm)]}
    landmarks: np.ndarray
    cameras: list              # list[Camera]


def make_stereo_cameras(cfg: SimConfig) -> list:
    cams = []
    offsets = [0.5 * cfg.baseline, -0.5 * cfg.baseline] if cfg.stereo else [0.0]
    for y in offsets:
        cams.append(Camera(cfg.width, cfg.height, cfg.focal, cfg.focal, cfg.width / 2 - 0.5,
                           cfg.height / 2 - 0.5, R_bc=R_BC_FORWARD.copy(),
                           p_bc=np.array([0.06, y, 0.0])))
    return cams


def make_landmarks(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    h, H = cfg.room_half, cfg.room_height
    pts = []
    n = cfg.n_wall_landmarks
    for _ in range(n):
        face = rng.integers(0, 6)
        u, v = rng.uniform(-h, h), rng.uniform(-h, h)
        z = rng.uniform(0, H)
        if face == 0:
            pts.append([h, u, z])
        elif face == 1:
            pts.append([-h, u, z])
        elif face == 2:
            pts.append([u, h, z])
        elif face == 3:
            pts.append([u, -h, z])
        elif face == 4:
            pts.append([u, v, 0.0])
        else:
            pts.append([u, v, H])
    # pillars give depth variety
    for k in range(cfg.n_pillar_landmarks):
        c = [(2.5, 2.5), (-2.5, 2.0), (2.0, -2.5), (-2.0, -2.8)][k % 4]
        ang = rng.uniform(0, 2 * np.pi)
        pts.append([c[0] + 0.3 * np.cos(ang), c[1] + 0.3 * np.sin(ang), rng.uniform(0, H)])
    return np.array(pts)


def simulate(cfg: SimConfig | None = None, traj: HeadTrajectory | None = None) -> SimData:
    cfg = cfg or SimConfig()
    traj = traj or HeadTrajectory()
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.imu_rate
    n = int(round(cfg.duration * cfg.imu_rate)) + 1
    t = np.arange(n) * dt
    P = np.array([traj.position(ti) for ti in t])
    V = np.array([traj.velocity(ti) for ti in t])
    Rs = np.array([traj.rotation(ti) for ti in t])

    nz = cfg.noise
    bg = np.zeros((n, 3))
    ba = np.zeros((n, 3))
    bg[0] = rng.normal(0, cfg.gyro_bias0_std, 3)
    ba[0] = rng.normal(0, cfg.accel_bias0_std, 3)
    for k in range(1, n):
        bg[k] = bg[k - 1] + rng.normal(0, nz.gyro_walk * np.sqrt(dt), 3)
        ba[k] = ba[k - 1] + rng.normal(0, nz.accel_walk * np.sqrt(dt), 3)

    gyro = np.zeros((n - 1, 3))
    accel = np.zeros((n - 1, 3))
    for k in range(n - 1):
        w = so3_log(Rs[k].T @ Rs[k + 1]) / dt
        a_w = (V[k + 1] - V[k]) / dt
        f = Rs[k].T @ (a_w - GRAVITY)
        gyro[k] = w + bg[k] + rng.normal(0, nz.gyro_noise / np.sqrt(dt), 3)
        accel[k] = f + ba[k] + rng.normal(0, nz.accel_noise / np.sqrt(dt), 3)

    # positions consistent with ZOH integration of the averaged accel (trapezoid on v)
    P_c = P.copy()
    for k in range(n - 1):
        P_c[k + 1] = P_c[k] + 0.5 * (V[k] + V[k + 1]) * dt

    cams = make_stereo_cameras(cfg)
    L = make_landmarks(cfg, rng)
    step = int(round(cfg.imu_rate / cfg.cam_rate))
    cam_idx = np.arange(0, n - 1, step)
    frames = []
    active: dict[int, list] = {}  # landmark -> [fid, remaining]
    next_fid = 0
    for k in cam_idx:
        R_wb, p_wb = Rs[k], P_c[k]
        vis = []
        for ci, cam in enumerate(cams):
            R_wc = R_wb @ cam.R_bc
            p_wc = p_wb + R_wb @ cam.p_bc
            pc = (L - p_wc) @ R_wc          # rows: R_wc^T (L - p_wc)
            uv, ok = cam.project(pc)
            ok &= pc[:, 2] > 0.2
            vis.append((uv, ok))
        seen_left = np.where(vis[0][1])[0]
        # continue active tracks, retire finished/invisible ones
        new_active = {}
        for lm in seen_left:
            if lm in active and active[lm][1] > 0:
                new_active[lm] = [active[lm][0], active[lm][1] - 1]
        candidates = [lm for lm in seen_left if lm not in new_active and lm not in active]
        rng.shuffle(candidates)
        for lm in candidates:
            if len(new_active) >= cfg.max_features:
                break
            new_active[lm] = [next_fid, int(rng.integers(*cfg.track_len_range))]
            next_fid += 1
        # landmarks that just finished a track stay out for one frame (then re-detected)
        active = new_active
        meas = {}
        for lm, (fid, _) in active.items():
            obs = []
            for ci, cam in enumerate(cams):
                uv, ok = vis[ci]
                if ok[lm]:
                    px = uv[lm] + rng.normal(0, cfg.pixel_noise, 2)
                    obs.append((ci, cam.pixel_to_normalized(px)[0]))
            if obs and obs[0][0] == 0:
                meas[fid] = obs
        frames.append(meas)

    return SimData(imu_t=t[:-1], gyro=gyro, accel=accel, gt_t=t, gt_p=P_c, gt_R=Rs, gt_v=V,
                   gt_bg=bg, gt_ba=ba, cam_t=t[cam_idx], cam_idx=cam_idx, frames=frames,
                   landmarks=L, cameras=cams)


def render_pattern_image(width: int, height: int, seed: int = 0, smooth: float = 2.0) -> np.ndarray:
    """Band-limited random texture (float32 in [0,255]) for front-end tests."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    filt = np.exp(-(fx ** 2 + fy ** 2) * (2 * np.pi * smooth) ** 2 / 2)
    img = np.real(np.fft.ifft2(np.fft.fft2(img) * filt))
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.float32)


def fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Exact sub-pixel translation of a periodic band-limited image: out(x) = img(x - d)."""
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    phase = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * phase)).astype(np.float32)


__all__ = ["SimConfig", "SimData", "HeadTrajectory", "simulate", "make_stereo_cameras",
           "render_pattern_image", "fourier_shift", "R_BC_FORWARD"]

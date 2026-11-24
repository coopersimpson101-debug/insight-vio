"""Glue: run the MSCKF on synthetic data (and helpers shared by scripts/tests)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .imu import Preintegrator, iterate_imu_segments
from .metrics import AteResult, ate
from .msckf import CameraExtrinsics, Msckf, MsckfConfig
from .sim import SimData


@dataclass
class RunResult:
    t: np.ndarray
    R: np.ndarray
    p: np.ndarray
    gt_R: np.ndarray
    gt_p: np.ndarray
    ate: AteResult
    seconds_per_frame: float
    stats: dict


def filter_config_for_sim(data: SimData, stereo: bool = True, pixel_sigma: float = 1.0,
                          **kw) -> MsckfConfig:
    cams = data.cameras if stereo else data.cameras[:1]
    return MsckfConfig(
        cameras=[CameraExtrinsics(c.R_bc, c.p_bc) for c in cams],
        obs_sigma=pixel_sigma / data.cameras[0].mean_focal, **kw)


def run_msckf_on_sim(data: SimData, cfg: MsckfConfig, stereo: bool = True,
                     init_sigma=None, outlier_frames: dict | None = None) -> RunResult:
    """Initialise from ground truth (with zero bias estimate) and run over all frames."""
    f = Msckf(cfg)
    k0 = data.cam_idx[0]
    sig = init_sigma if init_sigma is not None else \
        np.r_[[1e-3] * 3, [1e-3] * 3, [0.02] * 3, [0.02] * 3, [0.1] * 3]
    f.initialize(data.gt_t[k0], data.gt_R[k0], data.gt_p[k0], data.gt_v[k0],
                 bg=np.zeros(3), ba=np.zeros(3), P0=np.diag(np.asarray(sig) ** 2))
    imu_ptr = 0
    out_t, out_R, out_p = [], [], []
    t_start = time.perf_counter()
    for fi, (tc, meas) in enumerate(zip(data.cam_t, data.frames)):
        while imu_ptr < len(data.imu_t) and data.imu_t[imu_ptr] <= tc + 1e-9:
            f.add_imu(data.imu_t[imu_ptr], data.gyro[imu_ptr], data.accel[imu_ptr])
            imu_ptr += 1
        if not stereo:
            meas = {fid: [o for o in obs if o[0] == 0] for fid, obs in meas.items()}
        if outlier_frames and fi in outlier_frames:
            meas = dict(meas)
            meas.update(outlier_frames[fi])
        f.process_frame(tc, meas)
        t, R, p, _ = f.pose()
        out_t.append(t)
        out_R.append(R)
        out_p.append(p)
    elapsed = time.perf_counter() - t_start
    out_p = np.array(out_p)
    gt_p = data.gt_p[data.cam_idx]
    gt_R = data.gt_R[data.cam_idx]
    return RunResult(np.array(out_t), np.array(out_R), out_p, gt_R, gt_p, ate(out_p, gt_p),
                     elapsed / max(len(data.frames), 1), dict(f.stats))


def dead_reckon(data: SimData, bg=None, ba=None, gravity=np.array([0, 0, -9.81])):
    """IMU-only integration from ground-truth initial state (baseline for comparison)."""
    k0 = data.cam_idx[0]
    R, p, v = data.gt_R[k0], data.gt_p[k0], data.gt_v[k0]
    bg = np.zeros(3) if bg is None else bg
    ba = np.zeros(3) if ba is None else ba
    from .imu import ImuNoise
    out = [p]
    for a, b in zip(data.cam_t[:-1], data.cam_t[1:]):
        pre = Preintegrator(bg, ba, ImuNoise())
        for k, dt in iterate_imu_segments(data.imu_t, a, b):
            pre.integrate(data.gyro[k], data.accel[k], dt)
        R, p, v = pre.predict(R, p, v, gravity)
        out.append(p)
    return np.array(out)


def run_vio_on_rendered(data: SimData, stereo: bool = True, pixel_sigma: float = 1.5,
                        frontend_cfg=None, max_frames: int | None = None, renderer=None):
    """Full image pipeline (ray-cast images -> FAST/KLT front-end -> MSCKF) on synthetic data.

    Returns ``(RunResult, timings)`` where timings holds mean seconds per frame for
    rendering (not part of VIO), the front-end and the filter.
    """
    from .config import VioConfig
    from .frontend import FrontendConfig
    from .render import RoomRenderer
    from .system import VioSystem

    renderer = renderer or RoomRenderer()
    cams = data.cameras if stereo else data.cameras[:1]
    mcfg = filter_config_for_sim(data, stereo=stereo, pixel_sigma=pixel_sigma)
    cfg = VioConfig(cameras=cams, noise=mcfg.noise, time_offset=0.0, filter=mcfg,
                    frontend=frontend_cfg or FrontendConfig(max_features=120, fast_threshold=12))
    vio = VioSystem(cfg, stereo=stereo)
    k0 = data.cam_idx[0]
    sig = np.r_[[1e-3] * 3, [1e-3] * 3, [0.02] * 3, [0.02] * 3, [0.1] * 3]
    vio.initialize_from_state(data.gt_t[k0], data.gt_R[k0], data.gt_p[k0], data.gt_v[k0],
                              np.zeros(3), np.zeros(3), np.diag(sig ** 2))
    n = len(data.cam_t) if max_frames is None else min(max_frames, len(data.cam_t))
    imu_ptr = 0
    out_t, out_R, out_p = [], [], []
    t_render = t_vio = 0.0
    for fi in range(n):
        tc, k = data.cam_t[fi], data.cam_idx[fi]
        while imu_ptr < len(data.imu_t) and data.imu_t[imu_ptr] <= tc + 1e-9:
            vio.add_imu(data.imu_t[imu_ptr], data.gyro[imu_ptr], data.accel[imu_ptr])
            imu_ptr += 1
        a = time.perf_counter()
        imgs = []
        for cam in cams:
            R_wc = data.gt_R[k] @ cam.R_bc
            imgs.append(renderer.render(cam, R_wc, data.gt_p[k] + data.gt_R[k] @ cam.p_bc))
        b = time.perf_counter()
        t, R, p, _ = vio.process_images(tc, imgs[0], imgs[1] if stereo else None)
        t_vio += time.perf_counter() - b
        t_render += b - a
        out_t.append(t)
        out_R.append(R)
        out_p.append(p)
    out_p = np.array(out_p)
    gt_p = data.gt_p[data.cam_idx[:n]]
    gt_R = data.gt_R[data.cam_idx[:n]]
    res = RunResult(np.array(out_t), np.array(out_R), out_p, gt_R, gt_p, ate(out_p, gt_p),
                    t_vio / n, dict(vio.filter.stats))
    return res, {"render_s": t_render / n, "vio_s": t_vio / n}

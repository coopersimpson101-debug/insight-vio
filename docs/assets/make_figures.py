#!/usr/bin/env python
"""Regenerate every figure in the README from real runs of this repo.

    pip install matplotlib
    python docs/assets/make_figures.py            # ~2 min on a desktop CPU

Outputs (in docs/assets/):
  trajectory.png      top-down ground truth vs MSCKF vs IMU-only dead reckoning
  ate_over_time.png   per-frame position error of MSCKF and IMU-only integration
  frontend_tracks.png ray-cast stereo frame with FAST corners + KLT tracks overlaid
  banner.svg          README banner; the motif is the estimated trajectory path
  figures.txt         the numbers printed by the runs (for the README)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from insight_vio.config import VioConfig  # noqa: E402
from insight_vio.frontend import FrontendConfig  # noqa: E402
from insight_vio.metrics import ate  # noqa: E402
from insight_vio.pipeline import dead_reckon, filter_config_for_sim, run_msckf_on_sim  # noqa: E402
from insight_vio.render import RoomRenderer  # noqa: E402
from insight_vio.sim import SimConfig, simulate  # noqa: E402
from insight_vio.system import VioSystem  # noqa: E402

ACCENT = "#34D399"
BG, PANEL, FG, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
GT_C, DR_C = "#8b949e", "#f78166"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "monospace", "font.size": 10, "axes.titleweight": "bold",
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
})

log = []


def say(s):
    print(s)
    log.append(s)


def trajectory_figures(duration=30.0, seed=0):
    data = simulate(SimConfig(duration=duration, seed=seed, pixel_noise=0.5))
    cfg = filter_config_for_sim(data, stereo=True, pixel_sigma=1.0)
    res = run_msckf_on_sim(data, cfg, stereo=True)
    est = (res.ate.R @ res.p.T).T + res.ate.t          # SE(3)-aligned estimate
    gt = res.gt_p
    dr = dead_reckon(data)                             # same start, no alignment
    t = res.t - res.t[0]
    err_vio = np.linalg.norm(est - gt, axis=1)
    err_dr = np.linalg.norm(dr - gt, axis=1)
    say(f"stereo {duration:.0f} s seed {seed}: ATE rmse {res.ate.rmse * 100:.2f} cm, "
        f"max {res.ate.max * 100:.2f} cm, IMU-only ATE {ate(dr, gt).rmse:.2f} m, "
        f"IMU-only final error {err_dr[-1]:.2f} m")

    # --- top-down trajectory --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    pad = 0.35
    lo, hi = gt[:, :2].min(0) - pad, gt[:, :2].max(0) + pad
    ax.plot(dr[:, 0], dr[:, 1], color=DR_C, lw=1.2, alpha=0.8, label="IMU only (dead reckoning)")
    ax.plot(gt[:, 0], gt[:, 1], color=GT_C, lw=4.0, alpha=0.55, label="ground truth")
    ax.plot(est[:, 0], est[:, 1], color=ACCENT, lw=1.3, label="MSCKF estimate (stereo)")
    ax.plot(*gt[0, :2], "o", color=FG, ms=5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    # where does dead reckoning leave the frame?
    out = np.nonzero((dr[:, 0] < lo[0]) | (dr[:, 0] > hi[0]) |
                     (dr[:, 1] < lo[1]) | (dr[:, 1] > hi[1]))[0]
    if len(out):
        k = out[0]
        ax.annotate(f"IMU-only leaves the plot at t = {t[k]:.1f} s\n"
                    f"and ends {err_dr[-1]:.0f} m away",
                    xy=(np.clip(dr[k, 0], lo[0], hi[0]), np.clip(dr[k, 1], lo[1], hi[1])),
                    xytext=(0.03, 0.04), textcoords="axes fraction", color=DR_C, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=DR_C, lw=0.8))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Synthetic head motion, {duration:.0f} s, top-down view", loc="left")
    ax.text(0.99, 0.985, f"MSCKF ATE {res.ate.rmse * 100:.2f} cm RMSE",
            transform=ax.transAxes, ha="right", va="top", color=ACCENT, fontweight="bold")
    ax.grid(True, lw=0.5)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "trajectory.png"))
    plt.close(fig)

    # --- error over time -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=150)
    ax.semilogy(t, np.maximum(err_dr, 1e-4) * 100, color=DR_C, lw=1.3,
                label="IMU only (dead reckoning)")
    ax.semilogy(t, np.maximum(err_vio, 1e-4) * 100, color=ACCENT, lw=1.3,
                label="MSCKF stereo (SE(3)-aligned)")
    ax.axhline(res.ate.rmse * 100, color=ACCENT, lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [cm]")
    ax.set_title("Position error over time (same IMU data)", loc="left")
    ax.grid(True, which="major", lw=0.4)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ate_over_time.png"))
    plt.close(fig)
    return est


def frontend_figure(n_frames=24, seed=0):
    """Run the real image pipeline (ray-cast images -> VioSystem) and draw its tracks."""
    data = simulate(SimConfig(duration=n_frames / 20.0 + 0.2, seed=seed))
    renderer = RoomRenderer()
    cams = data.cameras
    mcfg = filter_config_for_sim(data, stereo=True, pixel_sigma=1.5)
    cfg = VioConfig(cameras=cams, noise=mcfg.noise, time_offset=0.0, filter=mcfg,
                    frontend=FrontendConfig(max_features=120, fast_threshold=12))
    vio = VioSystem(cfg, stereo=True)
    k0 = data.cam_idx[0]
    sig = np.r_[[1e-3] * 3, [1e-3] * 3, [0.02] * 3, [0.02] * 3, [0.1] * 3]
    vio.initialize_from_state(data.gt_t[k0], data.gt_R[k0], data.gt_p[k0], data.gt_v[k0],
                              np.zeros(3), np.zeros(3), np.diag(sig ** 2))
    hist = {}
    imu_ptr = 0
    for fi in range(n_frames):
        tc, k = data.cam_t[fi], data.cam_idx[fi]
        while imu_ptr < len(data.imu_t) and data.imu_t[imu_ptr] <= tc + 1e-9:
            vio.add_imu(data.imu_t[imu_ptr], data.gyro[imu_ptr], data.accel[imu_ptr])
            imu_ptr += 1
        imgs = [renderer.render(c, data.gt_R[k] @ c.R_bc, data.gt_p[k] + data.gt_R[k] @ c.p_bc)
                for c in cams]
        vio.process_images(tc, imgs[0], imgs[1])
        fe = vio.frontend
        alive = set(int(i) for i in fe.ids)
        for i, p in zip(fe.ids, fe.pts):
            hist.setdefault(int(i), []).append(p.copy())
        hist = {i: v for i, v in hist.items() if i in alive}
    left, right = imgs
    lens = [len(v) for v in hist.values()]
    say(f"front-end: {len(hist)} live tracks after {n_frames} frames, "
        f"median length {np.median(lens):.0f} frames, max {max(lens)}")

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=110,
                            gridspec_kw=dict(wspace=0.03))
    for ax, img, title in [(axs[0], left, "left camera: FAST-9 corners + KLT tracks"),
                           (axs[1], right, "right camera (stereo match target)")]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=10)
    for v in hist.values():
        v = np.array(v)
        if len(v) > 1:
            axs[0].plot(v[:, 0], v[:, 1], color=ACCENT, lw=1.1, alpha=0.9)
        axs[0].plot(v[-1, 0], v[-1, 1], "o", ms=3.2, mfc="none", mec=ACCENT, mew=1.0)
    fig.suptitle(f"Ray-cast synthetic room, frame {n_frames} @ 20 Hz, "
                 f"{len(hist)} tracked features (tails = last {n_frames} frames)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01)
    fig.savefig(os.path.join(HERE, "frontend_tracks.png"))
    plt.close(fig)


def banner(est):
    W, H = 1280, 400
    xy = est[:, :2]
    # fit the path into the right-hand box
    bx0, by0, bw, bh = 760, 40, 470, 320
    lo, hi = xy.min(0), xy.max(0)
    s = min(bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1]))
    cx, cy = bx0 + bw / 2, by0 + bh / 2
    mid = (lo + hi) / 2
    pts = [(cx + (x - mid[0]) * s, cy - (y - mid[1]) * s) for x, y in xy]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[::2])
    ex, ey = pts[-1]
    sx, sy = pts[0]
    ticks = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#34D399" opacity="0.9"/>'
                    for x, y in pts[::40])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="#e6edf3" stroke-opacity="0.05" stroke-width="1"/>
    </pattern>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#34D399" stop-opacity="0.16"/>
      <stop offset="1" stop-color="#34D399" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="fade" x1="0" x2="1">
      <stop offset="0" stop-color="#34D399" stop-opacity="0.15"/>
      <stop offset="1" stop-color="#34D399" stop-opacity="1"/>
    </linearGradient>
  </defs>
  <rect width="{W}" height="{H}" fill="#0d1117"/>
  <rect width="{W}" height="{H}" fill="url(#g)"/>
  <ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="330" ry="210" fill="url(#glow)"/>
  <polyline points="{path}" fill="none" stroke="url(#fade)" stroke-width="2.4"
            stroke-linejoin="round" stroke-linecap="round"/>
  {ticks}
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="none" stroke="#8b949e" stroke-width="1.5"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="7" fill="#34D399"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="14" fill="none" stroke="#34D399" stroke-opacity="0.5" stroke-width="1.5"/>
  <text x="{bx0 + bw:.0f}" y="{by0 + bh + 22:.0f}" text-anchor="end" fill="#8b949e" font-size="13"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">MSCKF estimate, synthetic run (top-down)</text>
  <g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
    <rect x="80" y="118" width="6" height="96" fill="#34D399"/>
    <text x="108" y="190" fill="#e6edf3" font-size="76" font-weight="700">insight-vio</text>
    <text x="110" y="248" fill="#8b949e" font-size="22">Stereo MSCKF visual-inertial odometry</text>
    <text x="110" y="280" fill="#8b949e" font-size="22">for a DIY inside-out VR headset</text>
    <text x="110" y="330" fill="#34D399" font-size="15">numpy · from scratch · FAST + KLT + MSCKF</text>
  </g>
</svg>
'''
    with open(os.path.join(HERE, "banner.svg"), "w", encoding="utf-8") as f:
        f.write(svg)


if __name__ == "__main__":
    est = trajectory_figures()
    banner(est)
    frontend_figure()
    with open(os.path.join(HERE, "figures.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

#!/usr/bin/env python
"""Micro-benchmarks of the Python pipeline stages (report the machine you ran it on!).

    python scripts/bench.py [--width 640 --height 480 --features 120]
"""
import argparse
import platform
import time

import _path  # noqa: F401
import numpy as np

from insight_vio.features import detect_features
from insight_vio.frontend import FrontendConfig, StereoFrontend
from insight_vio.imu import ImuNoise, Preintegrator
from insight_vio.klt import build_pyramid, track
from insight_vio.pipeline import filter_config_for_sim, run_msckf_on_sim
from insight_vio.render import RoomRenderer
from insight_vio.sim import SimConfig, fourier_shift, render_pattern_image, simulate


def timeit(fn, n=10):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--features", type=int, default=120)
    a = ap.parse_args()
    print(f"machine: {platform.processor() or platform.machine()} | {platform.system()} "
          f"{platform.release()} | Python {platform.python_version()} | numpy {np.__version__}")
    img = render_pattern_image(a.width, a.height, seed=0, smooth=3.0)
    img2 = fourier_shift(img, 3.2, -1.7)
    pts, _ = detect_features(img, a.features, 8)
    p0, p1 = build_pyramid(img), build_pyramid(img2)
    rows = [
        ("FAST+Shi-Tomasi detect", timeit(lambda: detect_features(img, a.features, 8))),
        ("pyramid build (4 lv)", timeit(lambda: build_pyramid(img))),
        (f"KLT track {len(pts)} pts", timeit(lambda: track(p0, p1, pts))),
    ]
    d = simulate(SimConfig(duration=4.0))
    cams = d.cameras
    rr = RoomRenderer()
    imgs = []
    for k in d.cam_idx[:12]:
        imgs.append([rr.render(c, d.gt_R[k] @ c.R_bc, d.gt_p[k] + d.gt_R[k] @ c.p_bc) for c in cams])
    fe = StereoFrontend(cams, FrontendConfig(max_features=a.features, fast_threshold=12))
    fe.process(*imgs[0])
    t = time.perf_counter()
    for l, r in imgs[1:]:
        fe.process(l, r)
    rows.append(("stereo front-end / frame", (time.perf_counter() - t) / (len(imgs) - 1) * 1e3))
    pre = Preintegrator(np.zeros(3), np.zeros(3), ImuNoise())
    rows.append(("preintegrate 1 IMU sample", timeit(lambda: pre.integrate(
        np.array([0.1, 0.2, 0.3]), np.array([0, 0, 9.8]), 0.005), 200)))
    res = run_msckf_on_sim(d, filter_config_for_sim(d, stereo=True), stereo=True)
    rows.append(("MSCKF back-end / frame", res.seconds_per_frame * 1e3))
    print(f"{'stage':32s} {'ms':>8s}")
    for name, ms in rows:
        print(f"{name:32s} {ms:8.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run insight-vio on a EuRoC sequence and write a TUM trajectory.

    python scripts/run_euroc.py data/euroc/MH_01_easy --out results/MH_01_easy.txt
    python scripts/evaluate.py results/MH_01_easy.txt data/euroc/MH_01_easy

Initialisation: by default from the ground-truth state at the first frame after
``--start`` seconds (isolates tracking quality); ``--static-init`` uses the
IMU-only gravity alignment instead (MH sequences start stationary).
Requires OpenCV for PNG decoding only (``pip install opencv-python-headless``).
"""
import argparse
import os
import time

import _path  # noqa: F401
import numpy as np

from insight_vio.config import load_config
from insight_vio.euroc import load_euroc, read_image
from insight_vio.lie import quat_to_rot
from insight_vio.metrics import save_tum
from insight_vio.msckf import CameraExtrinsics
from insight_vio.system import VioSystem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "euroc.yaml"))
    ap.add_argument("--out", default="estimate_tum.txt")
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--static-init", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="skip seconds at the start")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    seq = load_euroc(a.sequence, stereo=not a.mono)
    cfg = load_config(a.config)
    # calibration from the sequence itself overrides the config file
    cfg.cameras = seq.cameras
    cfg.noise = seq.imu_noise
    cfg.filter.noise = seq.imu_noise
    cfg.filter.cameras = [CameraExtrinsics(c.R_bc, c.p_bc) for c in seq.cameras]
    vio = VioSystem(cfg, stereo=not a.mono)

    t0 = seq.cam_t[0] + a.start
    imu_ptr = 0
    out_t, out_R, out_p = [], [], []
    tic = time.perf_counter()
    n = 0
    for tc, (fl, fr) in zip(seq.cam_t, seq.cam_files):
        if tc < t0:
            continue
        while imu_ptr < len(seq.imu_t) and seq.imu_t[imu_ptr] <= tc + cfg.time_offset:
            vio.add_imu(seq.imu_t[imu_ptr], seq.gyro[imu_ptr], seq.accel[imu_ptr])
            imu_ptr += 1
        if not vio.filter.initialized and not a.static_init:
            if seq.gt_t is None:
                raise SystemExit("no ground truth: use --static-init")
            g = int(np.argmin(np.abs(seq.gt_t - tc)))
            vio.initialize_from_state(tc, quat_to_rot(seq.gt_q[g]), seq.gt_p[g], seq.gt_v[g],
                                      seq.gt_bg[g], seq.gt_ba[g])
        left = read_image(fl)
        right = read_image(fr) if (fr and not a.mono) else None
        pose = vio.process_images(tc, left, right)
        if pose is not None:
            t, R, p, _ = pose
            out_t.append(t)
            out_R.append(R)
            out_p.append(p)
        n += 1
        if n % 100 == 0:
            el = time.perf_counter() - tic
            last = np.round(out_p[-1], 2) if out_p else "-"
            print(f"frame {n}: {el / n * 1e3:.0f} ms/frame, p = {last}")
        if a.max_frames and n >= a.max_frames:
            break
    el = time.perf_counter() - tic
    save_tum(a.out, out_t, out_R, out_p)
    print(f"wrote {len(out_t)} poses to {a.out}; {el / max(n, 1) * 1e3:.0f} ms/frame "
          f"(front-end + back-end, Python); filter stats {vio.filter.stats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the MSCKF on synthetic head motion and stream poses over UDP in (approximately)
real time - lets you develop the runtime side (Monado driver, viewer) without hardware.

    python scripts/live_viewer.py &          # open http://localhost:8080
    python examples/stream_synthetic.py --port 5555 --duration 30
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from insight_vio.msckf import Msckf  # noqa: E402
from insight_vio.pipeline import filter_config_for_sim  # noqa: E402
from insight_vio.sim import SimConfig, simulate  # noqa: E402
from insight_vio.udp import PoseSender  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--openxr", action="store_true")
    a = ap.parse_args()
    d = simulate(SimConfig(duration=a.duration))
    f = Msckf(filter_config_for_sim(d))
    k0 = d.cam_idx[0]
    f.initialize(d.gt_t[k0], d.gt_R[k0], d.gt_p[k0], d.gt_v[k0])
    tx = PoseSender(a.host, a.port, a.openxr)
    imu_ptr = 0
    start = time.perf_counter()
    for tc, meas in zip(d.cam_t, d.frames):
        while imu_ptr < len(d.imu_t) and d.imu_t[imu_ptr] <= tc:
            f.add_imu(d.imu_t[imu_ptr], d.gyro[imu_ptr], d.accel[imu_ptr])
            imu_ptr += 1
        f.process_frame(tc, meas)
        t, R, p, v = f.pose()
        w = d.gyro[max(imu_ptr - 1, 0)] - f.bg
        tx.send(int(t * 1e9), R, p, v, w, f.P[:6, :6])
        lag = (tc - d.cam_t[0]) - (time.perf_counter() - start)
        if lag > 0:
            time.sleep(lag)
    err = np.linalg.norm(p - d.gt_p[d.cam_idx[-1]])
    print(f"streamed {tx.seq} poses; final position error {err * 100:.1f} cm (unaligned)")


if __name__ == "__main__":
    main()

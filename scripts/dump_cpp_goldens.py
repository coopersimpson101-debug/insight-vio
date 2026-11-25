"""Dump IMU golden fixtures for the C++ preintegration tests.

Run from the repo root:
  python scripts/dump_cpp_goldens.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from insight_vio.imu import GRAVITY, ImuNoise, Preintegrator
from insight_vio.lie import so3_log
from insight_vio.sim import SimConfig, simulate


def noiseless_sim(duration=3.0, rate=200.0):
    n = ImuNoise(0.0, 0.0, 0.0, 0.0)
    cfg = SimConfig(duration=duration, imu_rate=rate, noise=n, gyro_bias0_std=0.0,
                    accel_bias0_std=0.0, n_wall_landmarks=10, n_pillar_landmarks=0)
    return simulate(cfg)


def write_matrix(path: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{' '.join(str(d) for d in arr.shape)}\n")
        if arr.ndim == 1:
            f.write(" ".join(f"{v:.17g}" for v in arr))
            f.write("\n")
        else:
            for row in arr:
                f.write(" ".join(f"{v:.17g}" for v in np.ravel(row)))
                f.write("\n")


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "..", "cpp", "tests", "goldens")
    os.makedirs(out, exist_ok=True)

    d = noiseless_sim()
    write_matrix(os.path.join(out, "imu_t.txt"), d.imu_t)
    write_matrix(os.path.join(out, "gt_t.txt"), d.gt_t)
    write_matrix(os.path.join(out, "gyro.txt"), d.gyro)
    write_matrix(os.path.join(out, "accel.txt"), d.accel)
    write_matrix(os.path.join(out, "gt_R.txt"), d.gt_R.reshape(len(d.gt_R), 9))
    write_matrix(os.path.join(out, "gt_p.txt"), d.gt_p)
    write_matrix(os.path.join(out, "gt_v.txt"), d.gt_v)
    write_matrix(os.path.join(out, "gravity.txt"), GRAVITY)

    # Reference preintegration on [40, 240) for regression checks.
    i, j = 40, 240
    pre = Preintegrator(np.zeros(3), np.zeros(3), ImuNoise())
    for k in range(i, j):
        pre.integrate(d.gyro[k], d.accel[k], d.imu_t[k + 1] - d.imu_t[k])
    write_matrix(os.path.join(out, "ref_dR.txt"), pre.dR)
    write_matrix(os.path.join(out, "ref_dv.txt"), pre.dv)
    write_matrix(os.path.join(out, "ref_dp.txt"), pre.dp)
    write_matrix(os.path.join(out, "ref_cov.txt"), pre.cov)
    write_matrix(os.path.join(out, "ref_J_Rg.txt"), pre.J_Rg)
    write_matrix(os.path.join(out, "ref_J_vg.txt"), pre.J_vg)
    write_matrix(os.path.join(out, "ref_J_va.txt"), pre.J_va)
    write_matrix(os.path.join(out, "ref_J_pg.txt"), pre.J_pg)
    write_matrix(os.path.join(out, "ref_J_pa.txt"), pre.J_pa)
    meta = np.array([i, j, pre.dt], dtype=np.float64)
    write_matrix(os.path.join(out, "ref_meta.txt"), meta)

    Ri, Rj = d.gt_R[i], d.gt_R[j]
    T = d.gt_t[j] - d.gt_t[i]
    err_R = so3_log(pre.dR.T @ (Ri.T @ Rj))
    print(f"wrote goldens to {os.path.abspath(out)}")
    print(f"ref window [{i},{j}) dt={pre.dt:.6f}s  |log err_R|={np.linalg.norm(err_R):.3e}")


if __name__ == "__main__":
    main()

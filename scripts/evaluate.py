#!/usr/bin/env python
"""ATE / RPE of a TUM trajectory against EuRoC ground truth (or another TUM file).

    python scripts/evaluate.py estimate.txt data/euroc/MH_01_easy [--sim3] [--delta 1.0]
"""
import argparse
import os

import _path  # noqa: F401
import numpy as np

from insight_vio.euroc import load_euroc
from insight_vio.lie import quat_to_rot
from insight_vio.metrics import associate, ate, interpolate_poses, load_tum, rpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate")
    ap.add_argument("groundtruth", help="EuRoC sequence folder or TUM file")
    ap.add_argument("--sim3", action="store_true", help="also estimate scale (mono VO)")
    ap.add_argument("--delta", type=float, default=1.0, help="RPE interval in seconds")
    ap.add_argument("--max-dt", type=float, default=0.02)
    a = ap.parse_args()

    t_e, p_e, q_e = load_tum(a.estimate)
    if os.path.isdir(a.groundtruth):
        seq = load_euroc(a.groundtruth, stereo=False)
        t_g, p_g, q_g = seq.gt_t, seq.gt_p, seq.gt_q
    else:
        t_g, p_g, q_g = load_tum(a.groundtruth)
    ie, _ = associate(t_e, t_g, a.max_dt)
    if len(ie) < 10:
        raise SystemExit("too few associated poses - check timestamps (seconds vs ns)")
    p_gi, R_gi = interpolate_poses(t_e[ie], t_g, p_g, q_g)
    R_e = np.array([quat_to_rot(q) for q in q_e[ie]])
    res = ate(p_e[ie], p_gi, with_scale=a.sim3)
    r = rpe(t_e[ie], R_e, p_e[ie], R_gi, p_gi, a.delta)
    length = float(np.sum(np.linalg.norm(np.diff(p_gi, axis=0), axis=1)))
    print(f"poses      : {len(ie)} associated, trajectory length {length:.1f} m")
    print(f"ATE (m)    : rmse {res.rmse:.4f}  mean {res.mean:.4f}  median {res.median:.4f}  "
          f"max {res.max:.4f}" + (f"  scale {res.scale:.4f}" if a.sim3 else ""))
    print(f"RPE ({a.delta:g} s): trans rmse {r.trans_rmse:.4f} m, rot rmse {r.rot_rmse_deg:.3f} deg "
          f"({r.n_pairs} pairs)")


if __name__ == "__main__":
    main()

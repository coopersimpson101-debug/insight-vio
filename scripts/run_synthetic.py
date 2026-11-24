#!/usr/bin/env python
"""Run the MSCKF on the synthetic head-motion dataset and report ATE / RPE.

    python scripts/run_synthetic.py --duration 30 --mono --out results/synthetic
"""
import argparse
import os

import _path  # noqa: F401
import numpy as np

from insight_vio.metrics import ate, rpe, save_tum
from insight_vio.pipeline import (dead_reckon, filter_config_for_sim, run_msckf_on_sim,
                                  run_vio_on_rendered)
from insight_vio.sim import SimConfig, simulate
from insight_vio.viewer import write_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mono", action="store_true", help="use only the left camera")
    ap.add_argument("--pixel-noise", type=float, default=0.5)
    ap.add_argument("--filter-sigma", type=float, default=None,
                    help="measurement sigma assumed by the filter in px (default max(1, 2*noise))")
    ap.add_argument("--max-clones", type=int, default=11)
    ap.add_argument("--images", action="store_true",
                    help="ray-cast stereo images and run the real FAST/KLT front-end "
                         "(slow) instead of feeding simulated feature tracks")
    ap.add_argument("--out", default=None, help="directory for trajectory + HTML viewer")
    a = ap.parse_args()

    data = simulate(SimConfig(duration=a.duration, seed=a.seed, pixel_noise=a.pixel_noise))
    stereo = not a.mono
    sigma = a.filter_sigma if a.filter_sigma else max(1.0, 2.0 * a.pixel_noise)
    cfg = filter_config_for_sim(data, stereo=stereo, pixel_sigma=sigma, max_clones=a.max_clones)
    if a.images:
        res, timing = run_vio_on_rendered(data, stereo=stereo)
    else:
        res = run_msckf_on_sim(data, cfg, stereo=stereo)
    r = rpe(res.t, res.R, res.p, res.gt_R, res.gt_p, delta=1.0)
    dr = ate(dead_reckon(data), res.gt_p)
    feats = np.mean([len(f) for f in data.frames])
    src = "rendered images + FAST/KLT front-end" if a.images else         f"simulated tracks ({feats:.0f}/frame, {a.pixel_noise} px noise, filter sigma {sigma} px)"
    print(f"sequence : synthetic head motion, {a.duration:.0f} s, seed {a.seed}, "
          f"{'stereo' if stereo else 'mono'}, {src}")
    print(f"ATE      : rmse {res.ate.rmse * 100:.2f} cm | mean {res.ate.mean * 100:.2f} cm | "
          f"max {res.ate.max * 100:.2f} cm  (SE3 Umeyama alignment)")
    print(f"RPE(1 s) : {r.trans_rmse * 100:.2f} cm, {r.rot_rmse_deg:.3f} deg")
    print(f"IMU only : ATE rmse {dr.rmse:.2f} m")
    what = "front-end + back-end" if a.images else "back-end only"
    print(f"timing   : {res.seconds_per_frame * 1e3:.1f} ms/frame ({what}, this machine)")
    print(f"stats    : {res.stats}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        save_tum(os.path.join(a.out, "estimate.txt"), res.t, res.R, res.p)
        save_tum(os.path.join(a.out, "groundtruth.txt"), res.t, res.gt_R, res.gt_p)
        aligned = (res.ate.R @ res.p.T).T + res.ate.t
        write_html(os.path.join(a.out, "trajectory.html"),
                   {"MSCKF (aligned)": aligned, "ground truth": res.gt_p},
                   "insight-vio synthetic", f"ATE rmse {res.ate.rmse * 100:.2f} cm")
        print(f"wrote {a.out}/estimate.txt, groundtruth.txt, trajectory.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Render TUM trajectories into a self-contained interactive HTML 3D viewer.

    python scripts/view_trajectory.py estimate.txt groundtruth.txt -o traj.html --align
"""
import argparse

import _path  # noqa: F401

from insight_vio.metrics import associate, ate, load_tum
from insight_vio.viewer import write_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories", nargs="+")
    ap.add_argument("-o", "--out", default="trajectory.html")
    ap.add_argument("--align", action="store_true", help="align the first onto the second (SE3)")
    a = ap.parse_args()
    trajs = {path: load_tum(path) for path in a.trajectories}
    shown, stat = {}, ""
    names = list(trajs)
    if a.align and len(names) >= 2:
        (te, pe, _), (tg, pg, _) = trajs[names[0]], trajs[names[1]]
        ie, ig = associate(te, tg)
        r = ate(pe[ie], pg[ig])
        shown[names[0] + " (aligned)"] = (r.R @ pe.T).T + r.t
        stat = f"ATE rmse {r.rmse:.3f} m"
        names = names[1:]
    for n in names:
        shown[n] = trajs[n][1]
    write_html(a.out, shown, "insight-vio", stat)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

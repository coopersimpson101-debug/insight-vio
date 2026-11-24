#!/usr/bin/env python
"""Convert Kalibr output into an insight-vio config.

    python scripts/kalibr_to_config.py camchain-imucam.yaml imu.yaml \
        --base configs/pi5_ov9281_bmi088.yaml -o configs/my_headset.yaml
"""
import argparse

import _path  # noqa: F401
import yaml

from insight_vio.config import kalibr_to_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("camchain")
    ap.add_argument("imu", nargs="?")
    ap.add_argument("--base", help="config to take filter/frontend/hardware sections from")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    with open(a.camchain) as f:
        cc = yaml.safe_load(f)
    imu = None
    if a.imu:
        with open(a.imu) as f:
            imu = yaml.safe_load(f)
    out = {}
    if a.base:
        with open(a.base) as f:
            out = yaml.safe_load(f)
    out.update(kalibr_to_config(cc, imu))
    with open(a.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"wrote {a.out} ({len(out['cameras'])} cameras, time_offset {out['time_offset']:+.4f} s)")


if __name__ == "__main__":
    main()

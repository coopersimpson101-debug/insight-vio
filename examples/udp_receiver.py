#!/usr/bin/env python
"""Minimal consumer of the insight-vio UDP pose stream (what a runtime driver does).

    python examples/udp_receiver.py --port 5555
    # in another terminal, stream a synthetic run:
    python examples/stream_synthetic.py --port 5555
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insight_vio.udp import FLAG_OPENXR, FLAG_VALID, receive_forever  # noqa: E402


def show(p):
    axes = "OpenXR" if p.flags & FLAG_OPENXR else "VIO"
    state = "OK  " if p.flags & FLAG_VALID else "INIT"
    x, y, z = p.position
    w, qx, qy, qz = p.quat_wxyz
    print(f"#{p.seq:6d} t={p.t_ns * 1e-9:12.4f}s [{state} {axes}] "
          f"p=({x:+.3f},{y:+.3f},{z:+.3f}) q=({w:+.3f},{qx:+.3f},{qy:+.3f},{qz:+.3f}) "
          f"sigma_p={p.pos_std * 100:.1f}cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to listen on; 0.0.0.0 to accept poses from the LAN")
    a = ap.parse_args()
    receive_forever(a.port, a.host, callback=show)

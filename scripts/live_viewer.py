#!/usr/bin/env python
"""Live web viewer for the UDP pose stream: open http://<host>:8080 in a browser.

    python scripts/live_viewer.py --udp-port 5555 --http-port 8080
"""
import argparse

import _path  # noqa: F401

from insight_vio.viewer import serve_live

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int, default=5555)
    ap.add_argument("--http-port", type=int, default=8080)
    a = ap.parse_args()
    serve_live(a.http_port, a.udp_port)

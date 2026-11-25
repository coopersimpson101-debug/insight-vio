#!/usr/bin/env python
"""Live web viewer for the UDP pose stream: open http://<host>:8080 in a browser.

    python scripts/live_viewer.py --udp-port 5555 --http-port 8080
    python scripts/live_viewer.py --host 0.0.0.0   # accept poses from a headset on the LAN
"""
import argparse

import _path  # noqa: F401

from insight_vio.viewer import serve_live

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int, default=5555)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface for both UDP and HTTP; 0.0.0.0 to expose them on the LAN")
    a = ap.parse_args()
    serve_live(a.http_port, a.udp_port, host=a.host)

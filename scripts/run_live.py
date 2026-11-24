#!/usr/bin/env python
"""Live headset tracking: cameras + SPI IMU -> VIO -> UDP pose stream.

**UNTESTED ON HARDWARE** - this is the wiring the author still has to validate
(see README "Hardware validation / TODO").  Typical use on a Pi 5:

    sudo apt install python3-picamera2 python3-spidev python3-libgpiod
    python scripts/run_live.py --config configs/my_headset.yaml --record /tmp/session1

``--record`` stores an EuRoC-style folder (cam0/cam1 PNGs + imu0/data.csv) so
sessions can be replayed offline with scripts/run_euroc.py --static-init.
"""
import argparse
import os
import queue
import threading
import time

import _path  # noqa: F401
import numpy as np

from insight_vio.config import load_config
from insight_vio.system import VioSystem
from insight_vio.udp import FLAG_DEGRADED, FLAG_INITIALIZING, FLAG_VALID, PoseSender


def make_imu(hw):
    from insight_vio.hw.imu_spi import BMI088, ICM42688, SpidevBus
    if hw.get("imu_driver", "bmi088") == "bmi088":
        imu = BMI088(SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs_accel", 0)),
                     SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs_gyro", 1)))
    else:
        imu = ICM42688(SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs", 0)))
    imu.configure()
    return imu


def make_cameras(hw):
    from insight_vio.hw import cameras
    if hw.get("camera_driver") == "picamera2":
        src = cameras.Picamera2Stereo(hw.get("width", 640), hw.get("height", 400),
                                      hw.get("fps", 30), hw.get("exposure_us", 3000))
        src.start()
        return src
    return cameras.OpenCVStereo(cameras.jetson_csi_pipeline(0, hw.get("width", 1280),
                                                            hw.get("height", 800), hw.get("fps", 30)),
                                cameras.jetson_csi_pipeline(1, hw.get("width", 1280),
                                                            hw.get("height", 800), hw.get("fps", 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", default=None)
    ap.add_argument("--imu-rate", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    hw = cfg.hardware
    from insight_vio.hw.imu_spi import ImuReader
    imu = make_imu(hw)
    rate = a.imu_rate or 400.0
    reader = ImuReader(imu, rate, hw.get("R_body_imu"))
    imu_q: "queue.Queue" = queue.Queue()
    threading.Thread(target=reader.run, args=(lambda t, w, a_: imu_q.put((t, w, a_)),),
                     daemon=True).start()
    cams = make_cameras(hw)
    vio = VioSystem(cfg, stereo=len(cfg.cameras) > 1)
    out = cfg.output
    sender = PoseSender(out.get("udp_host", "127.0.0.1"), out.get("udp_port", 5555),
                        out.get("openxr_axes", True))
    rec = None
    if a.record:
        import cv2
        for d in ("cam0/data", "cam1/data", "imu0"):
            os.makedirs(os.path.join(a.record, "mav0", d), exist_ok=True)
        rec = open(os.path.join(a.record, "mav0", "imu0", "data.csv"), "w")
        rec.write("#timestamp [ns],wx,wy,wz,ax,ay,az\n")
        cam_csv = [open(os.path.join(a.record, "mav0", c, "data.csv"), "w") for c in ("cam0", "cam1")]
    last_w = np.zeros(3)
    print("waiting for the headset to be still for static initialisation ...")
    for t_ns, left, right in cams.frames():
        while not imu_q.empty():
            ti, w, acc = imu_q.get()
            vio.add_imu(ti * 1e-9, w, acc)
            last_w = w
            if rec:
                rec.write(f"{ti},{w[0]},{w[1]},{w[2]},{acc[0]},{acc[1]},{acc[2]}\n")
        if rec:
            for ci, im in enumerate((left, right)):
                if im is not None:
                    cv2.imwrite(os.path.join(a.record, "mav0", f"cam{ci}", "data", f"{t_ns}.png"), im)
                    cam_csv[ci].write(f"{t_ns},{t_ns}.png\n")
        tic = time.perf_counter()
        pose = vio.process_images(t_ns * 1e-9, left, right)
        dt_ms = (time.perf_counter() - tic) * 1e3
        if pose is None:
            sender.send(t_ns, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3),
                        flags=FLAG_INITIALIZING)
            continue
        t, R, p, v = pose
        flags = FLAG_VALID | (FLAG_DEGRADED if dt_ms > 100 else 0)
        sender.send(int(t * 1e9), R, p, v, last_w - vio.filter.bg, vio.filter.P[:6, :6], flags)


if __name__ == "__main__":
    main()

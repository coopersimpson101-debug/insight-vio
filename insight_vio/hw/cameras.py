"""Camera sources for live operation.  **Not validated on hardware yet.**

* :class:`Picamera2Stereo` - Raspberry Pi 5 with two OV9281 global-shutter mono
  modules on CAM0/CAM1 via libcamera/picamera2.  For true simultaneous exposure
  wire the modules' external trigger (XVS/FSIN, module-specific - e.g. Arducam's
  "external trigger" OV9281 boards) to a shared GPIO/PWM signal, or put one sensor
  in master and one in slave mode if the driver supports it.  Without hardware
  sync, frames are paired by nearest ``SensorTimestamp`` (expect up to half a frame
  of skew, which hurts stereo at fast head rotations).
* :class:`OpenCVStereo` - Jetson Orin Nano (GStreamer ``nvarguscamerasrc`` pipeline
  strings) or USB UVC stereo cameras.  OpenCV is used here **for camera I/O only**.

Each source yields ``(t_ns, left uint8 HxW, right uint8 HxW or None)`` with ``t_ns``
on the host monotonic clock (mid-exposure if the driver reports exposure time).
"""
from __future__ import annotations

import time

import numpy as np


class Picamera2Stereo:  # pragma: no cover - needs Pi 5 + cameras
    def __init__(self, width: int = 1280, height: int = 800, fps: float = 30.0,
                 exposure_us: int = 4000, gain: float = 2.0, max_pair_skew_ns: int = 2_000_000):
        from picamera2 import Picamera2
        self.cams = [Picamera2(0), Picamera2(1)]
        frame_us = int(1e6 / fps)
        for c in self.cams:
            cfg = c.create_video_configuration(
                main={"size": (width, height), "format": "YUV420"},   # Y plane = mono
                controls={"FrameDurationLimits": (frame_us, frame_us),
                          "ExposureTime": exposure_us, "AnalogueGain": gain,
                          "AeEnable": False}, buffer_count=4)
            c.configure(cfg)
        self.h, self.w = height, width
        self.max_skew = max_pair_skew_ns

    def start(self):
        for c in self.cams:
            c.start()

    def _grab(self, cam):
        req = cam.capture_request()
        try:
            y = req.make_array("main")[: self.h, : self.w].copy()
            md = req.get_metadata()
        finally:
            req.release()
        # SensorTimestamp: start of exposure of the first line, ns, CLOCK_BOOTTIME on
        # recent libcamera builds; shift to mid-exposure for a global shutter.
        t = int(md["SensorTimestamp"]) + int(md.get("ExposureTime", 0)) * 500
        return t, y

    def frames(self):
        while True:
            tl, l = self._grab(self.cams[0])
            tr, r = self._grab(self.cams[1])
            while tr < tl - self.max_skew:      # resynchronise by dropping the older frame
                tr, r = self._grab(self.cams[1])
            while tl < tr - self.max_skew:
                tl, l = self._grab(self.cams[0])
            yield (tl + tr) // 2, l, r

    def stop(self):
        for c in self.cams:
            c.stop()


def jetson_csi_pipeline(sensor_id: int, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    """GStreamer pipeline for a CSI camera on Jetson (Argus), output GRAY8."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM),width={width},"
            f"height={height},framerate={fps}/1 ! nvvidconv ! video/x-raw,format=GRAY8 ! "
            f"appsink drop=1 max-buffers=2")


class OpenCVStereo:  # pragma: no cover - needs cameras
    def __init__(self, left, right=None):
        import cv2
        self.cv2 = cv2
        api = cv2.CAP_GSTREAMER if isinstance(left, str) else cv2.CAP_ANY
        self.caps = [cv2.VideoCapture(left, api)]
        if right is not None:
            self.caps.append(cv2.VideoCapture(right, api))

    def frames(self):
        while True:
            for c in self.caps:
                c.grab()                       # grab both first to minimise skew
            t = time.monotonic_ns()            # host timestamp: calibrate td with Kalibr!
            imgs = []
            for c in self.caps:
                ok, im = c.retrieve()
                if not ok:
                    return
                if im.ndim == 3:
                    im = self.cv2.cvtColor(im, self.cv2.COLOR_BGR2GRAY)
                imgs.append(np.ascontiguousarray(im))
            yield t, imgs[0], (imgs[1] if len(imgs) > 1 else None)

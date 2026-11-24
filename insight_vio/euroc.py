"""EuRoC MAV dataset (ASL format) loader.

Expected layout (as extracted from the official zips)::

    <seq>/mav0/cam0/data.csv, cam0/data/<ns>.png, cam0/sensor.yaml
    <seq>/mav0/cam1/...
    <seq>/mav0/imu0/data.csv, imu0/sensor.yaml
    <seq>/mav0/state_groundtruth_estimate0/data.csv

Timestamps are converted from nanoseconds to seconds (float64, relative values
are what matter).  The body frame ``B`` is the IMU frame ``S`` of ``imu0``
(its ``T_BS`` is identity), cameras' ``T_BS`` give ``(R_bc, p_bc)``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

from .camera import Camera
from .imu import ImuNoise
from .lie import quat_to_rot


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # EuRoC files start with an OpenCV-style "%YAML:1.0" directive PyYAML rejects
    lines = [l for l in text.splitlines() if not l.startswith("%YAML")]
    return yaml.safe_load("\n".join(lines))


def _read_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", comments="#", ndmin=2, dtype=np.float64)


def load_camera(sensor_yaml: str) -> Camera:
    d = _read_yaml(sensor_yaml)
    T = np.array(d["T_BS"]["data"], dtype=float).reshape(4, 4)
    fu, fv, cu, cv = d["intrinsics"]
    model = d.get("distortion_model", "radial-tangential")
    model = {"radial-tangential": "radtan", "radtan": "radtan", "equidistant": "equidistant",
             "none": "none"}[model]
    w, h = d["resolution"]
    return Camera(int(w), int(h), fu, fv, cu, cv, model, list(d.get("distortion_coefficients", [])),
                  R_bc=T[:3, :3], p_bc=T[:3, 3])


def load_imu_noise(sensor_yaml: str) -> ImuNoise:
    d = _read_yaml(sensor_yaml)
    return ImuNoise(gyro_noise=float(d["gyroscope_noise_density"]),
                    accel_noise=float(d["accelerometer_noise_density"]),
                    gyro_walk=float(d["gyroscope_random_walk"]),
                    accel_walk=float(d["accelerometer_random_walk"]))


@dataclass
class EurocSequence:
    root: str
    imu_t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    cam_t: np.ndarray
    cam_files: list            # list of (left_path, right_path or None)
    cameras: list
    imu_noise: ImuNoise
    gt_t: np.ndarray | None
    gt_p: np.ndarray | None
    gt_q: np.ndarray | None    # Hamilton wxyz
    gt_v: np.ndarray | None
    gt_bg: np.ndarray | None
    gt_ba: np.ndarray | None

    def gt_rotations(self) -> np.ndarray:
        return np.array([quat_to_rot(q) for q in self.gt_q])


def load_euroc(path: str, stereo: bool = True) -> EurocSequence:
    mav = os.path.join(path, "mav0") if os.path.isdir(os.path.join(path, "mav0")) else path
    imu = _read_csv(os.path.join(mav, "imu0", "data.csv"))
    imu_t = imu[:, 0] * 1e-9
    gyro, accel = imu[:, 1:4], imu[:, 4:7]

    def cam_list(name):
        rows = []
        with open(os.path.join(mav, name, "data.csv"), "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                ts, fn = line.strip().split(",")[:2]
                rows.append((int(ts), os.path.join(mav, name, "data", fn.strip())))
        return rows

    left = cam_list("cam0")
    cams = [load_camera(os.path.join(mav, "cam0", "sensor.yaml"))]
    files = [(fn, None) for _, fn in left]
    if stereo and os.path.isdir(os.path.join(mav, "cam1")):
        right = dict(cam_list("cam1"))
        cams.append(load_camera(os.path.join(mav, "cam1", "sensor.yaml")))
        files = [(fn, right.get(ts)) for ts, fn in left]
    cam_t = np.array([ts for ts, _ in left], dtype=np.float64) * 1e-9

    noise_path = os.path.join(mav, "imu0", "sensor.yaml")
    noise = load_imu_noise(noise_path) if os.path.exists(noise_path) else ImuNoise()

    gt_path = os.path.join(mav, "state_groundtruth_estimate0", "data.csv")
    gt = _read_csv(gt_path) if os.path.exists(gt_path) else None
    if gt is not None:
        g = dict(gt_t=gt[:, 0] * 1e-9, gt_p=gt[:, 1:4], gt_q=gt[:, 4:8], gt_v=gt[:, 8:11],
                 gt_bg=gt[:, 11:14], gt_ba=gt[:, 14:17])
    else:
        g = dict(gt_t=None, gt_p=None, gt_q=None, gt_v=None, gt_bg=None, gt_ba=None)
    return EurocSequence(path, imu_t, gyro, accel, cam_t, files, cams, noise, **g)


def read_image(path: str) -> np.ndarray:
    """Decode a grayscale PNG (OpenCV used for decoding only)."""
    import cv2  # optional dependency: image I/O only
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

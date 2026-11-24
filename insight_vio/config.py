"""YAML configuration: cameras, IMU noise, extrinsics, time offset, filter and front-end.

See ``configs/*.yaml`` for annotated examples.  ``T_imu_cam`` is the 4x4 transform
taking camera coordinates to IMU/body coordinates (``p_b = T_imu_cam p_c``), i.e.
the inverse of Kalibr's ``T_cam_imu``.  The time offset follows Kalibr:
``t_imu = t_cam + time_offset``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import yaml

from .camera import Camera
from .frontend import FrontendConfig
from .imu import ImuNoise
from .msckf import CameraExtrinsics, MsckfConfig


@dataclass
class VioConfig:
    cameras: list
    noise: ImuNoise
    time_offset: float
    filter: MsckfConfig
    frontend: FrontendConfig
    hardware: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)


def _camera_from_dict(d: dict) -> Camera:
    fx, fy, cx, cy = d["intrinsics"]
    w, h = d["resolution"]
    T = np.array(d.get("T_imu_cam", np.eye(4).tolist()), dtype=float).reshape(4, 4)
    return Camera(int(w), int(h), float(fx), float(fy), float(cx), float(cy),
                  d.get("distortion_model", "none"), list(d.get("distortion", [])),
                  R_bc=T[:3, :3], p_bc=T[:3, 3])


def config_from_dict(d: dict) -> VioConfig:
    cams = [_camera_from_dict(c) for c in d["cameras"]]
    imu = d.get("imu", {})
    noise = ImuNoise(gyro_noise=float(imu.get("gyro_noise", ImuNoise.gyro_noise)),
                     accel_noise=float(imu.get("accel_noise", ImuNoise.accel_noise)),
                     gyro_walk=float(imu.get("gyro_walk", ImuNoise.gyro_walk)),
                     accel_walk=float(imu.get("accel_walk", ImuNoise.accel_walk)))
    td = float(d.get("time_offset", 0.0))
    f = d.get("filter", {})
    pixel_sigma = float(f.get("pixel_sigma", 1.0))
    mcfg = MsckfConfig(
        cameras=[CameraExtrinsics(c.R_bc, c.p_bc) for c in cams],
        noise=noise,
        max_clones=int(f.get("max_clones", 11)),
        obs_sigma=pixel_sigma / cams[0].mean_focal,
        chi2_prob=float(f.get("chi2_prob", 0.95)),
        chi2_multiplier=float(f.get("chi2_multiplier", 1.0)),
        time_offset=td,
        max_depth=float(f.get("max_depth", 60.0)),
    )
    fe = d.get("frontend", {})
    fcfg = FrontendConfig(**{k: v for k, v in fe.items() if k in FrontendConfig.__dataclass_fields__})
    return VioConfig(cams, noise, td, mcfg, fcfg, d.get("hardware", {}), d.get("output", {}))


def load_config(path: str) -> VioConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(yaml.safe_load(fh))


def kalibr_to_config(camchain: dict, imu: dict | None = None) -> dict:
    """Convert Kalibr ``camchain-imucam.yaml`` (+ optional ``imu.yaml``) into our config dict."""
    cams = []
    td = 0.0
    for key in sorted(k for k in camchain if k.startswith("cam")):
        c = camchain[key]
        T_cam_imu = np.array(c["T_cam_imu"], dtype=float)
        T_imu_cam = np.linalg.inv(T_cam_imu)
        dist = {"radtan": "radtan", "equidistant": "equidistant", "none": "none"}[
            c.get("distortion_model", "radtan")]
        cams.append({
            "name": key, "model": c.get("camera_model", "pinhole"),
            "distortion_model": dist,
            "intrinsics": [float(x) for x in c["intrinsics"]],
            "distortion": [float(x) for x in c.get("distortion_coeffs", [])],
            "resolution": [int(x) for x in c["resolution"]],
            "T_imu_cam": T_imu_cam.round(9).tolist(),
        })
        if key == "cam0":
            td = float(c.get("timeshift_cam_imu", 0.0))
    out = {"cameras": cams, "time_offset": td}
    if imu:
        i = imu.get("imu0", imu)
        out["imu"] = {"gyro_noise": float(i["gyroscope_noise_density"]),
                      "accel_noise": float(i["accelerometer_noise_density"]),
                      "gyro_walk": float(i["gyroscope_random_walk"]),
                      "accel_walk": float(i["accelerometer_random_walk"])}
    return out

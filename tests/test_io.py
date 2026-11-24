import os
import socket
import struct

import numpy as np
import pytest
import yaml

from insight_vio.config import config_from_dict, kalibr_to_config, load_config
from insight_vio.euroc import load_euroc
from insight_vio.hw.imu_spi import BMI088, ICM42688, MockSpiBus
from insight_vio.hw.timesync import LinearClockModel, estimate_time_offset
from insight_vio.lie import rot_to_quat, so3_exp
from insight_vio.metrics import load_tum, save_tum
from insight_vio.udp import (FLAG_OPENXR, FLAG_VALID, PACKET_SIZE, PosePacket, PoseSender, pack,
                             to_openxr, unpack)
from insight_vio.viewer import render_html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EUROC_CAM_YAML = """%YAML:1.0
sensor_type: camera
comment: VI-Sensor cam0 (MT9M034)
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375]
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
"""

EUROC_IMU_YAML = """%YAML:1.0
sensor_type: imu
T_BS:
  cols: 4
  rows: 4
  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
rate_hz: 200
gyroscope_noise_density: 1.6968e-04
gyroscope_random_walk: 1.9393e-05
accelerometer_noise_density: 2.0000e-3
accelerometer_random_walk: 3.0000e-3
"""


def make_fake_euroc(tmp_path):
    mav = tmp_path / "MH_01_easy" / "mav0"
    for d in ("cam0/data", "cam1/data", "imu0", "state_groundtruth_estimate0"):
        (mav / d).mkdir(parents=True)
    t0 = 1403636579758555392
    for cam in ("cam0", "cam1"):
        (mav / cam / "sensor.yaml").write_text(EUROC_CAM_YAML)
        with open(mav / cam / "data.csv", "w") as f:
            f.write("#timestamp [ns],filename\n")
            for k in range(5):
                ts = t0 + k * 50_000_000
                f.write(f"{ts},{ts}.png\n")
    (mav / "imu0" / "sensor.yaml").write_text(EUROC_IMU_YAML)
    with open(mav / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for k in range(50):
            f.write(f"{t0 + k * 5_000_000},0.01,-0.02,0.03,9.7,0.1,-0.2\n")
    with open(mav / "state_groundtruth_estimate0" / "data.csv", "w") as f:
        f.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z\n")
        for k in range(50):
            f.write(f"{t0 + k * 5_000_000},{0.01 * k},0,1,1,0,0,0,0.2,0,0,0,0,0,0,0,0\n")
    return mav.parent


def test_euroc_loader(tmp_path):
    seq = load_euroc(str(make_fake_euroc(tmp_path)))
    assert len(seq.cam_t) == 5 and len(seq.imu_t) == 50
    assert np.isclose(seq.cam_t[1] - seq.cam_t[0], 0.05)
    assert seq.cameras[0].distortion_model == "radtan" and len(seq.cameras) == 2
    assert np.isclose(np.linalg.det(seq.cameras[0].R_bc), 1.0, atol=1e-6)
    assert np.isclose(seq.imu_noise.gyro_noise, 1.6968e-4)
    assert seq.cam_files[0][1] is not None and seq.cam_files[0][0].endswith(".png")
    assert seq.gt_p.shape == (50, 3) and np.allclose(seq.gt_rotations()[0], np.eye(3))


@pytest.mark.parametrize("name", ["euroc.yaml", "pi5_ov9281_bmi088.yaml",
                                  "jetson_orin_icm42688.yaml", "synthetic.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(os.path.join(ROOT, "configs", name))
    assert len(cfg.cameras) >= 1
    for c in cfg.cameras:
        assert np.allclose(c.R_bc.T @ c.R_bc, np.eye(3), atol=1e-6)
    assert cfg.filter.obs_sigma > 0


def test_kalibr_conversion_roundtrip():
    T_cam_imu = np.eye(4)
    T_cam_imu[:3, :3] = so3_exp(np.array([0.01, -1.57, 0.02]))
    T_cam_imu[:3, 3] = [0.03, -0.01, 0.02]
    camchain = {"cam0": {"T_cam_imu": T_cam_imu.tolist(), "camera_model": "pinhole",
                         "distortion_model": "equidistant", "distortion_coeffs": [0.1, 0, 0, 0],
                         "intrinsics": [280.0, 281.0, 320.0, 200.0], "resolution": [640, 400],
                         "timeshift_cam_imu": 0.0042}}
    imu = {"imu0": {"gyroscope_noise_density": 1e-3, "accelerometer_noise_density": 1e-2,
                    "gyroscope_random_walk": 1e-5, "accelerometer_random_walk": 1e-4}}
    d = kalibr_to_config(camchain, imu)
    cfg = config_from_dict(yaml.safe_load(yaml.safe_dump(d)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cfg.cameras[0].R_bc, cfg.cameras[0].p_bc
    assert np.allclose(T @ T_cam_imu, np.eye(4), atol=1e-8)
    assert cfg.time_offset == pytest.approx(0.0042) and cfg.filter.time_offset == pytest.approx(0.0042)
    assert cfg.noise.gyro_noise == pytest.approx(1e-3)


def test_udp_packet_roundtrip_and_crc():
    R = so3_exp(np.array([0.2, -0.1, 0.7]))
    pkt = PosePacket(7, 123456789012345, np.array([1.0, 2.0, 3.0]), rot_to_quat(R),
                     np.array([0.1, 0.2, 0.3]), np.array([-0.5, 0.0, 0.5]), 0.01, 0.002, FLAG_VALID)
    buf = pack(pkt)
    assert len(buf) == PACKET_SIZE == 84
    assert buf[:4] == b"IVIO"
    out = unpack(buf)
    assert out.seq == 7 and out.t_ns == 123456789012345
    assert np.allclose(out.position, pkt.position, atol=1e-6)
    assert np.allclose(out.quat_wxyz, pkt.quat_wxyz, atol=1e-6)
    assert out.pos_std == pytest.approx(0.01)
    corrupted = bytearray(buf)
    corrupted[25] ^= 0xFF
    with pytest.raises(ValueError):
        unpack(bytes(corrupted))
    # documented layout: position at offset 20
    assert struct.unpack_from("<3f", buf, 20) == pytest.approx((1.0, 2.0, 3.0))


def test_udp_send_receive_loopback():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]
    tx = PoseSender("127.0.0.1", port, openxr_axes=True)
    P = np.diag([1e-4] * 3 + [4e-4] * 3)
    tx.send(1000, np.eye(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), P)
    pkt = unpack(rx.recvfrom(256)[0])
    tx.close()
    rx.close()
    assert pkt.flags & FLAG_OPENXR
    # 1 m forward in VIO (x) is -Z in OpenXR
    assert np.allclose(pkt.position, [0, 0, -1.0], atol=1e-6)
    assert pkt.pos_std == pytest.approx(0.02, rel=1e-5)


def test_openxr_axes_conversion():
    # gravity (world -z in VIO) must become -Y in OpenXR; looking forward -> -Z
    R, p, v, w = to_openxr(np.eye(3), np.zeros(3), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]))
    assert np.allclose(v, [0, -1, 0]) and np.allclose(w, [0, 0, -1])
    assert np.isclose(np.linalg.det(R), 1.0)


def test_bmi088_decoding_with_mock_bus():
    acc_regs = {0x00: 0x1E}
    gyr_regs = {0x00: 0x0F}
    # accel raw x=+16384 (range 6g -> 0.5*6g*... see datasheet formula), gyro x=+16384 (1000 dps)
    acc_regs.update(dict(zip(range(0x12, 0x18), struct.pack("<hhh", 16384, -8192, 0))))
    gyr_regs.update(dict(zip(range(0x02, 0x08), struct.pack("<hhh", 16384, 0, -16384))))
    imu = BMI088(MockSpiBus(acc_regs, dummy_byte=True), MockSpiBus(gyr_regs), acc_range=0x01,
                 gyr_range=0x00)
    imu.configure(sleep=lambda s: None)
    w, a = imu.read()
    assert np.allclose(np.degrees(w), [1000, 0, -1000])
    # 16384 LSB at +-6 g range: 16384/32768 * 2^(1+1) * 1.5 g = 3 g
    assert np.allclose(a / 9.80665, [3.0, -1.5, 0.0])
    with pytest.raises(IOError):
        BMI088(MockSpiBus({0x00: 0x00}, dummy_byte=True), MockSpiBus(gyr_regs)).configure(
            sleep=lambda s: None)


def test_icm42688_decoding_with_mock_bus():
    regs = {0x75: 0x47}
    regs.update(dict(zip(range(0x1F, 0x2B), struct.pack(">hhhhhh", 4096, 0, -4096, 164, 0, 0))))
    imu = ICM42688(MockSpiBus(regs), gyro_fs=0, accel_fs=1)
    imu.configure(sleep=lambda s: None)
    w, a = imu.read()
    assert np.allclose(a / 9.80665, [1.0, 0.0, -1.0])        # 8 g range -> 4096 LSB/g
    assert np.allclose(np.degrees(w)[0], 164 * 2000 / 32768)
    assert (0x4E, 0x0F) in imu.bus.writes


def test_linear_clock_model_and_time_offset():
    rng = np.random.default_rng(0)
    m = LinearClockModel(forgetting=1.0)
    for k in range(2000):
        ts = k * 0.0025
        m.update(ts, 100.0 + ts * (1 + 50e-6) + rng.normal(0, 50e-6))
    assert abs(m.drift_ppm - 50) < 5
    assert abs(m.to_host(10.0) - (100 + 10 * (1 + 50e-6))) < 2e-5
    # cross-correlation of angular-speed signals
    t_imu = np.arange(0, 20, 0.005)
    speed = lambda t: 1.0 + 0.8 * np.sin(1.3 * t) + 0.5 * np.sin(3.7 * t + 1) + 0.3 * np.sin(7.1 * t)
    td = 0.0137
    t_cam = np.arange(0.5, 19.5, 0.05)
    cam_speed = speed(t_cam + td) + rng.normal(0, 0.02, len(t_cam))
    est = estimate_time_offset(t_imu, speed(t_imu), t_cam, cam_speed)
    assert abs(est - td) < 1e-3


def test_tum_io_and_viewer(tmp_path):
    t = np.arange(5) * 0.1
    Rs = [so3_exp(np.array([0, 0, 0.1 * k])) for k in range(5)]
    ps = [np.array([k, 0.0, 1.0]) for k in range(5)]
    path = tmp_path / "traj.txt"
    save_tum(path, t, Rs, ps)
    t2, p2, q2 = load_tum(path)
    assert np.allclose(t2, t) and np.allclose(p2, ps) and np.allclose(q2[1], rot_to_quat(Rs[1]))
    html = render_html({"est": np.array(ps), "gt": np.array(ps)}, "test")
    assert "<canvas" in html and '"name": "est"' in html

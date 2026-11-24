# Calibration workflow (Kalibr)

VIO accuracy is dominated by calibration. Budget an afternoon.

## 0. Record tooling

Kalibr consumes ROS bags. The easiest path without ROS on the headset:

1. Record an EuRoC-style folder on the headset:
   `python scripts/run_live.py --config configs/pi5_ov9281_bmi088.yaml --record ~/calib/static`
   (VIO output can be ignored during calibration recordings.)
2. Convert to a bag on a desktop with Kalibr's Docker image:
   `rosrun kalibr kalibr_bagcreater --folder ~/calib/dyn/mav0 --output-bag dyn.bag`
   (`kalibr_bagcreater` expects `cam0/`, `cam1/`, `imu0.csv` - rename `imu0/data.csv`
   accordingly; check the script's expected layout for your Kalibr version.)

## 1. IMU noise (Allan variance)

Leave the headset perfectly still for 2-3 hours, record IMU only, then run
`allan_variance_ros` (or any Allan-deviation tool) to get

* `gyroscope_noise_density`, `accelerometer_noise_density` (white noise, slope −1/2)
* `gyroscope_random_walk`, `accelerometer_random_walk` (slope +1/2)

Inflate them (x2 to x10 is common practice) before passing them to Kalibr and to
`configs/*.yaml` - unmodelled vibration and temperature effects are real.

## 2. Camera intrinsics + stereo extrinsics

* Target: Aprilgrid (6x6, print flat on rigid board, measure tag size precisely).
* Move the **target** slowly in front of the static rig (or vice versa), covering all
  image regions, incl. corners, at several distances and tilts. 60-90 s at 20 Hz.
* ```
  kalibr_calibrate_cameras --bag static.bag --target aprilgrid.yaml \
      --models pinhole-equi pinhole-equi --topics /cam0/image_raw /cam1/image_raw
  ```
* Accept when reprojection error is < 0.3 px (mono global shutter, good focus).
  Use `pinhole-equi` for > 90° lenses, `pinhole-radtan` for narrow lenses.

## 3. Camera-IMU extrinsics + time offset

* Keep the target static, move the **rig**: excite all 6 axes (roll/pitch/yaw rotations,
  x/y/z translations), ~1 minute, smooth but brisk, keep the target in view, avoid
  motion blur (short exposure, good light).
* ```
  kalibr_calibrate_imu_camera --bag dyn.bag --cam camchain.yaml --imu imu.yaml \
      --target aprilgrid.yaml
  ```
* Output: `camchain-imucam.yaml` with `T_cam_imu` and `timeshift_cam_imu` per camera.
* Sanity checks: translation between cameras matches the ruler-measured baseline within a
  couple of mm; `timeshift_cam_imu` is a few ms and repeatable across two recordings.

## 4. Import into insight-vio

```bash
python scripts/kalibr_to_config.py camchain-imucam.yaml imu.yaml \
    --base configs/pi5_ov9281_bmi088.yaml -o configs/my_headset.yaml
```

Conventions handled by the converter:

| Kalibr | insight-vio |
|---|---|
| `T_cam_imu` (IMU → camera) | `T_imu_cam` = inverse (camera → IMU/body) |
| `timeshift_cam_imu`, `t_imu = t_cam + shift` | `time_offset` (same sign) |
| `pinhole-radtan`, `pinhole-equi` | `radtan`, `equidistant` |

## 5. Verify

* Replay a recording: `python scripts/run_euroc.py ~/sessions/s1 --static-init`.
* Cross-check the time offset with `insight_vio.hw.timesync.estimate_time_offset`.
* Walk a closed loop (start = end) and check the end-point drift; record it in the README
  table.

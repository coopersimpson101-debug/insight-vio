# Hardware guide (Pi 5 or Jetson Orin Nano)

> **Status:** the drivers in `insight_vio/hw/` are written against datasheets and
> library docs and their decoding logic is unit-tested with mock SPI buses. **None of it
> has been run on real hardware yet.** Treat this page as a build plan.

## Reference build A - Raspberry Pi 5

| part | notes |
|---|---|
| Raspberry Pi 5 (4 or 8 GB) | two 4-lane MIPI CSI/DSI connectors (CAM/DISP 0 and 1) |
| 2x OV9281 1MP global-shutter **mono** modules | global shutter avoids rolling-shutter distortion during fast head turns; mono gives more light per pixel. Pick modules with an external trigger (FSIN/XVS) pin if you want hardware sync |
| 2x M12 lenses, ~100-130° diagonal FOV | wide FOV = more features and better rotation conditioning; calibrate with `pinhole-equi` |
| BMI088 (or ICM-42688-P) breakout | BMI088 has good vibration robustness; ICM-42688-P has lower noise |
| rigid mount | **cameras and IMU must be on one stiff plate** - any flex breaks the fixed extrinsic assumption |

Mount the cameras forward-facing with a baseline of 6-10 cm, ideally slightly
toed-out/tilted down to see more of the room (a VR user looks at the floor and walls).

### Wiring (BMI088 on SPI0)

| BMI088 pin | Pi 5 header |
|---|---|
| VDD / VDDIO | 3V3 (pin 1) |
| GND | GND (pin 6) |
| SCK | GPIO11 SCLK (pin 23) |
| SDI (MOSI) | GPIO10 MOSI (pin 19) |
| SDO1 + SDO2 (MISO, tie together) | GPIO9 MISO (pin 21) |
| CSB1 (accel CS) | GPIO8 CE0 (pin 24) |
| CSB2 (gyro CS) | GPIO7 CE1 (pin 26) |
| INT3 (gyro data-ready) | GPIO23 (pin 16) |

Enable SPI: `dtparam=spi=on` in `/boot/firmware/config.txt`. Run SPI at <= 10 MHz.

### Camera configuration

```bash
# /boot/firmware/config.txt
camera_auto_detect=0
dtoverlay=ov9281,cam0
dtoverlay=ov9281,cam1
```

`insight_vio/hw/cameras.py::Picamera2Stereo` configures both sensors with fixed exposure
(short: 2-4 ms to limit motion blur) and fixed frame duration, and pairs frames by
`SensorTimestamp`.

### Camera-camera hardware sync

Software pairing leaves up to half a frame of skew between left and right, which
directly corrupts stereo depth during head rotation (at 200°/s, 5 ms skew = 1°).
Options, best first:

1. External trigger: drive both sensors' FSIN/XVS input from one Pi GPIO (hardware PWM
   at the frame rate) with the sensors configured for external-trigger / slave mode.
   Requires modules + driver support (check your vendor's OV9281 driver; Arducam ships
   trigger-capable boards). Set `hardware.hardware_sync: true`.
2. Master/slave: one sensor outputs XVS, the other is slaved.
3. No sync: keep fps high and rely on the `max_pair_skew_ns` check.

### Camera-IMU time sync

* Timestamp IMU samples in the data-ready interrupt with the kernel GPIO event
  timestamp (libgpiod v2 edge events), not after the SPI read.
* `SensorTimestamp` from libcamera is the start of exposure; `cameras.py` shifts it to
  mid-exposure (global shutter).
* Verify both use the same clock (`CLOCK_MONOTONIC` vs `CLOCK_BOOTTIME` differ only after
  suspend, but check).
* The remaining constant offset is estimated by Kalibr (`timeshift_cam_imu` ->
  `time_offset`). Cross-check with `hw.timesync.estimate_time_offset` (gyro vs visual
  rotation-rate cross-correlation).
* If the IMU has an internal counter (BMI088 accel `SENSORTIME`), fit it to host time
  with `hw.timesync.LinearClockModel` to remove user-space jitter.

## Reference build B - Jetson Orin Nano

Same cameras (CSI, via `nvarguscamerasrc` + `jetson_csi_pipeline()`), ICM-42688-P on
SPI1 (enable with `sudo /opt/nvidia/jetson-io/jetson-io.py`). The Orin's CPU cores are
faster than the Pi 5's, and the C++ port could move FAST/KLT to CUDA or VPI.

## Running (after calibration)

```bash
python scripts/kalibr_to_config.py camchain-imucam.yaml imu.yaml \
    --base configs/pi5_ov9281_bmi088.yaml -o configs/my_headset.yaml
python scripts/run_live.py --config configs/my_headset.yaml --record ~/sessions/s1
python scripts/live_viewer.py          # on the Pi, then browse to http://<pi>:8080
```

`--record` writes an EuRoC-format folder that can be replayed with
`scripts/run_euroc.py --static-init`.

## Performance expectations

The Python pipeline needs roughly 0.4-0.5 s per stereo frame on a desktop Ryzen 5 3600
(see README), so it is **not real-time on a Pi 5**. Use the Python version for
recording sessions, offline replay and algorithm development; real-time headset
tracking needs the C++ port (`docs/cpp_port.md`).

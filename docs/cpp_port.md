# C++ port plan

The Python code is structured so each module maps to one C++ translation unit with
Eigen for the small dense algebra. Nothing below exists yet.

| Python module | C++ target | notes |
|---|---|---|
| `lie.py` | `lie.hpp` (header-only) | `Eigen::Matrix3d`, keep right-perturbation convention; port `tests/test_lie.py` 1:1 to GoogleTest |
| `imu.py` | `preintegration.{hpp,cpp}` | fixed-size `Matrix<double,9,9>`; ~0.14 ms/sample in Python -> < 1 µs expected |
| `features.py` | `fast.cpp` | FAST with the 4-compass early-rejection test, NEON (Pi 5 / Orin are Cortex-A76 / A78AE) |
| `klt.py` | `klt.cpp` | per-point loops are cache friendly; precompute template + gradient once per level; NEON bilinear; or OpenCV `calcOpticalFlowPyrLK` as a reference to compare against |
| `frontend.py` | `frontend.cpp` | run left/right tracking on two threads |
| `triangulation.py` | `triangulation.cpp` | 3x3 normal equations, unrolled |
| `msckf.py` | `msckf.{hpp,cpp}` | dynamic `MatrixXd` for P (max 15 + 6*11 = 81); Householder QR for null-space and compression; only touch the needed covariance blocks |
| `udp.py` | `pose_udp.cpp` | packed struct (see `docs/udp_protocol.md`) |
| `hw/imu_spi.py` | `imu_spi.cpp` | `spidev` ioctl + libgpiod v2 edge events with kernel timestamps |
| `hw/cameras.py` | `camera_libcamera.cpp` | libcamera C++ API directly (no picamera2) |

## Threading model

```
IMU thread (1 kHz, RT priority)  ->  lock-free ring buffer  --+
camera thread(s) (30 Hz)  ->  front-end thread  ->  filter thread  ->  UDP
                                                   +-> IMU-rate propagator -> UDP (high-rate pose)
```

## Expected budget on Pi 5 (to be measured)

| stage | Python, Ryzen 5 3600 (measured) | C++ Pi 5 (estimate, unverified) |
|---|---:|---:|
| FAST + Shi-Tomasi, 640x480 | 118 ms | 1-3 ms |
| KLT, 120 points, 4 levels | 115 ms | 1-2 ms |
| stereo front-end per frame | 400 ms | 4-8 ms |
| MSCKF update | 60-75 ms | 1-3 ms |

The C++ numbers are rough expectations from comparable open-source systems
(OpenVINS, MSCKF-VIO, Basalt report a few ms per frame on desktop CPUs) and must be
replaced with measurements.

## Validation strategy

* Golden tests: dump inputs/outputs of every Python module (`np.save`) on the synthetic
  sequence, assert the C++ version matches to 1e-9 (math) / 1e-3 px (KLT).
* The end-to-end ATE tests from `tests/test_msckf.py` run against the C++ library through
  pybind11 bindings.

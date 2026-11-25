# C++ port plan

The Python code is structured so each module maps to one C++ translation unit with
Eigen for the small dense algebra. Sources live under [`cpp/`](../cpp/).

| Python module | C++ target | status |
|---|---|---|
| `lie.py` | `include/insight_vio/lie.hpp` (header-only) | done — GoogleTest port of `tests/test_lie.py` |
| `imu.py` | `preintegration.{hpp,cpp}` | done — golden fixtures from the synthetic IMU + Jacobian tests |
| `triangulation.py` | `triangulation.{hpp,cpp}` | done — exact / noisy / behind-camera GoogleTests |
| `features.py` | `fast.cpp` | not started — FAST with 4-compass early rejection, NEON |
| `klt.py` | `klt.cpp` | not started — template+gradient precompute; NEON bilinear or OpenCV reference |
| `frontend.py` | `frontend.cpp` | not started — left/right tracking on two threads |
| `msckf.py` | `msckf.{hpp,cpp}` | not started — `MatrixXd` P, Householder QR null-space / compression |
| `udp.py` | `pose_udp.cpp` | not started — packed struct (see `docs/udp_protocol.md`) |
| `hw/imu_spi.py` | `imu_spi.cpp` | not started — `spidev` + libgpiod v2 |
| `hw/cameras.py` | `camera_libcamera.cpp` | not started — libcamera C++ API |

## Build and test

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/insight_vio_microbench   # optional; prints measured us/call on this machine
```

Eigen 3.4 and GoogleTest are fetched by CMake (`FetchContent`). Regenerate IMU
goldens after changing the Python simulator or preintegrator:

```bash
python scripts/dump_cpp_goldens.py
```

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
| SO(3) exp+log / IMU integrate (desktop C++, see microbench) | — | measured locally; do not invent Pi numbers |

The C++ frame-budget numbers are rough expectations from comparable open-source systems
(OpenVINS, MSCKF-VIO, Basalt report a few ms per frame on desktop CPUs) and must be
replaced with measurements. The microbench only times the math already ported.

## Validation strategy

* Golden tests: dump inputs/outputs of every Python module (`scripts/dump_cpp_goldens.py`)
  on the synthetic sequence, assert the C++ version matches to 1e-9 (math) / 1e-3 px (KLT).
* Port unit tests 1:1 where practical (Lie, triangulation).
* The end-to-end ATE tests from `tests/test_msckf.py` will eventually run against the C++
  library through pybind11 bindings (not started).

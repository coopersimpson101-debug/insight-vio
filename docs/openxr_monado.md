# OpenXR / Monado integration notes

Goal: a DIY headset (display + Pi/Jetson running insight-vio) shows up in OpenXR apps
as a 6DoF HMD. **Nothing on this page has been built or tested yet.** It is a plan.

## Option 1 - Monado driver that reads the UDP stream (recommended)

[Monado](https://monado.freedesktop.org/) is the open-source OpenXR runtime. Its drivers
implement `struct xrt_device`; the HMD driver's `get_tracked_pose` callback returns an
`xrt_space_relation` (pose + velocities + validity flags) for a requested time.

Sketch of a `drivers/insight_vio/` driver:

1. Background thread: `recvfrom()` on UDP port 5555, validate magic/version/CRC (see
   `docs/udp_protocol.md`), keep the latest packet under a mutex.
2. Send poses with `openxr_axes: true` so no axis conversion is needed in the driver.
3. `get_tracked_pose(dev, name, at_timestamp_ns, out_relation)`:
   * `dt = at_timestamp_ns - pkt.t_ns` (both monotonic; if the VIO runs on a separate
     computer, estimate the clock offset, e.g. with an NTP-style ping on a second port).
   * predict: orientation `q ⊗ Exp(ω dt)`, position `p + v dt` (clamp `dt` to ~50 ms).
   * set `XRT_SPACE_RELATION_*_VALID_BIT` and `*_TRACKED_BIT` from flags bit0.
4. Register the driver through a builder/prober and enable it with an environment
   variable or `config_v0.json`, following an existing simple driver as template (look
   for the "remote" / "simulated" drivers in the Monado tree; APIs change between
   versions, so copy the current structure rather than this sketch).

Display, distortion and lens parameters are separate (Monado's HMD distortion config).

## Option 2 - Monado's own SLAM tracker interface

Monado has a generic SLAM tracking interface (`t_slam`, used with Basalt / ORB-SLAM3 /
Kimera via its "SLAM tracker" adapter). A C++ port of insight-vio could implement that
interface directly, receiving camera frames and IMU samples from Monado instead of from
its own drivers - this removes the UDP hop and gives Monado's built-in pose prediction
and filtering. Requires the C++ port.

## Option 3 - SteamVR

A SteamVR OpenVR driver (`ITrackedDeviceServerDriver::GetPose`) can consume the same UDP
packets; SteamVR expects Y-up, −Z forward (same as OpenXR axes) and accepts velocity
for its own prediction.

## Coordinate conventions checklist

* Quaternion order in the packet is **w, x, y, z** (Hamilton). Monado's `xrt_quat` is
  stored `x, y, z, w` - reorder!
* Position in metres.
* Head frame origin: insight-vio reports the **IMU** pose. Apply the fixed IMU→eye-center
  transform (from your CAD model) in the driver.
* Recenter: OpenXR apps assume the floor is at y=0 for STAGE space; VIO starts at
  z=0 (or wherever initialised). Add a "recenter / floor height" offset in the driver.

## Latency budget (target)

| stage | target |
|---|---|
| exposure mid-point -> frame in memory | 10-15 ms |
| VIO update (C++ port) | < 10 ms |
| UDP + driver | < 1 ms |
| prediction to photon time | handled by runtime |

The Python implementation (~0.4 s/frame on a desktop CPU) cannot meet this.

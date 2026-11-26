# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- C++17 + Eigen scaffold under `cpp/`: header-only `lie.hpp`, Forster preintegration,
  triangulation, GoogleTests with IMU golden fixtures, and a CI `cpp` job

## [0.1.0] - 2026-11-25

First public release of a hand-written MSCKF visual-inertial odometry stack for inside-out VR headset tracking.

### Added
- Lie-group utilities, FAST + Shi-Tomasi corners, pyramidal KLT, stereo tracking
- Forster IMU preintegration with bias Jacobians; MSCKF with null-space projection and chi-squared gating
- Synthetic head-motion simulator: 0.50 cm ATE over 30 s of stereo (IMU-only dead reckoning: 25.9 m)
- BMI088 and ICM-42688 drivers, Pi 5 and Jetson camera sources, 84-byte UDP pose packet

[0.1.0]: https://github.com/coopersimpson101-debug/insight-vio/releases/tag/v0.1.0

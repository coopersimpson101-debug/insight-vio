// Micro-benchmark for the C++ lie + preintegration path.
// Reports numbers measured on this machine; do not invent Pi 5 figures.

#include <insight_vio/lie.hpp>
#include <insight_vio/preintegration.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

namespace {

template <typename Fn>
double time_ms(Fn&& fn, int repeats) {
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    fn();
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

}  // namespace

int main() {
  std::mt19937_64 rng(0);
  std::normal_distribution<double> n(0.0, 1.0);

  volatile double sink = 0.0;
  const double so3_ms = time_ms(
      [&] {
        Eigen::Vector3d phi(n(rng), n(rng), n(rng));
        phi *= 0.5;
        const Eigen::Matrix3d R = insight_vio::so3_exp(phi);
        sink += insight_vio::so3_log(R).norm();
      },
      200000);

  insight_vio::Preintegrator pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                 insight_vio::ImuNoise{});
  const double imu_us = 1000.0 * time_ms(
                                     [&] {
                                       Eigen::Vector3d w(n(rng) * 0.1, n(rng) * 0.1, n(rng) * 0.1);
                                       Eigen::Vector3d a(n(rng), n(rng), 9.81 + n(rng) * 0.1);
                                       pre.integrate(w, a, 0.005);
                                     },
                                     50000);

  std::printf("insight-vio C++ microbench\n");
  std::printf("  so3_exp+so3_log: %.4f us/call\n", so3_ms * 1000.0);
  std::printf("  Preintegrator::integrate: %.4f us/sample\n", imu_us);
  std::printf("  sink=%.6g (anti-DCE)\n", sink + pre.dt);
  return 0;
}

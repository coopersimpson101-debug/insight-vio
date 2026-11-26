#pragma once
// On-manifold IMU preintegration (Forster et al., T-RO 2017). Matches insight_vio/imu.py.

#include <insight_vio/lie.hpp>

#include <Eigen/Dense>

#include <tuple>
#include <utility>
#include <vector>

namespace insight_vio {

inline const Eigen::Vector3d kGravity(0.0, 0.0, -9.81);

struct ImuNoise {
  double gyro_noise = 1.6968e-4;   // rad/s/sqrt(Hz)
  double accel_noise = 2.0e-3;     // m/s^2/sqrt(Hz)
  double gyro_walk = 1.9393e-5;    // rad/s^2/sqrt(Hz)
  double accel_walk = 3.0e-3;      // m/s^3/sqrt(Hz)
};

class Preintegrator {
 public:
  Preintegrator(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, ImuNoise noise);

  void integrate(const Eigen::Vector3d& w_m, const Eigen::Vector3d& a_m, double dt);

  // First-order bias correction without re-integration (Forster eq. 44).
  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> corrected(
      const Eigen::Vector3d& bg_new, const Eigen::Vector3d& ba_new) const;

  std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> predict(
      const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
      const Eigen::Vector3d& g = kGravity) const;

  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  ImuNoise noise;
  Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
  Eigen::Vector3d dv = Eigen::Vector3d::Zero();
  Eigen::Vector3d dp = Eigen::Vector3d::Zero();
  double dt = 0.0;
  Eigen::Matrix<double, 9, 9> cov = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix3d J_Rg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_vg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_va = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_pg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_pa = Eigen::Matrix3d::Zero();
};

// Yield (index, dt) for zero-order-hold IMU samples covering [t0, t1).
std::vector<std::pair<int, double>> iterate_imu_segments(const Eigen::VectorXd& times,
                                                         double t0, double t1);

}  // namespace insight_vio

#include <insight_vio/preintegration.hpp>

#include <algorithm>
#include <cmath>

namespace insight_vio {

Preintegrator::Preintegrator(const Eigen::Vector3d& bg_in, const Eigen::Vector3d& ba_in,
                             ImuNoise noise_in)
    : bg(bg_in), ba(ba_in), noise(std::move(noise_in)) {}

void Preintegrator::integrate(const Eigen::Vector3d& w_m, const Eigen::Vector3d& a_m, double dt_s) {
  if (dt_s <= 0.0) {
    return;
  }
  const Eigen::Vector3d w = w_m - bg;
  const Eigen::Vector3d a = a_m - ba;
  const Eigen::Matrix3d dR_prev = dR;
  const Eigen::Matrix3d a_sk = skew(a);
  const Eigen::Matrix3d dRk = so3_exp(w * dt_s);
  const Eigen::Matrix3d Jr = so3_right_jacobian(w * dt_s);
  const double dt2 = dt_s * dt_s;

  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
  A.block<3, 3>(0, 0) = dRk.transpose();
  A.block<3, 3>(3, 0) = -dR_prev * a_sk * dt_s;
  A.block<3, 3>(6, 0) = -0.5 * dR_prev * a_sk * dt2;
  A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt_s;

  Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
  B.block<3, 3>(0, 0) = Jr * dt_s;
  B.block<3, 3>(3, 3) = dR_prev * dt_s;
  B.block<3, 3>(6, 3) = 0.5 * dR_prev * dt2;

  const double qg = (noise.gyro_noise * noise.gyro_noise) / dt_s;
  const double qa = (noise.accel_noise * noise.accel_noise) / dt_s;
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  Q.diagonal() << qg, qg, qg, qa, qa, qa;
  cov = A * cov * A.transpose() + B * Q * B.transpose();

  J_pa = J_pa + J_va * dt_s - 0.5 * dR_prev * dt2;
  J_pg = J_pg + J_vg * dt_s - 0.5 * dR_prev * a_sk * J_Rg * dt2;
  J_va = J_va - dR_prev * dt_s;
  J_vg = J_vg - dR_prev * a_sk * J_Rg * dt_s;
  J_Rg = dRk.transpose() * J_Rg - Jr * dt_s;

  const Eigen::Vector3d Ra = dR_prev * a;
  dp = dp + dv * dt_s + 0.5 * Ra * dt2;
  dv = dv + Ra * dt_s;
  dR = dR_prev * dRk;
  dt += dt_s;
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> Preintegrator::corrected(
    const Eigen::Vector3d& bg_new, const Eigen::Vector3d& ba_new) const {
  const Eigen::Vector3d dbg = bg_new - bg;
  const Eigen::Vector3d dba = ba_new - ba;
  const Eigen::Matrix3d dR_c = dR * so3_exp(J_Rg * dbg);
  const Eigen::Vector3d dv_c = dv + J_vg * dbg + J_va * dba;
  const Eigen::Vector3d dp_c = dp + J_pg * dbg + J_pa * dba;
  return {dR_c, dv_c, dp_c};
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector3d> Preintegrator::predict(
    const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
    const Eigen::Vector3d& g) const {
  const double T = dt;
  const Eigen::Matrix3d R_j = R_i * dR;
  const Eigen::Vector3d v_j = v_i + g * T + R_i * dv;
  const Eigen::Vector3d p_j = p_i + v_i * T + 0.5 * g * T * T + R_i * dp;
  return {R_j, p_j, v_j};
}

std::vector<std::pair<int, double>> iterate_imu_segments(const Eigen::VectorXd& times, double t0,
                                                         double t1) {
  std::vector<std::pair<int, double>> out;
  if (t1 <= t0) {
    return out;
  }
  const int n = static_cast<int>(times.size());
  // searchsorted(times, t0, side="right") - 1
  int k = static_cast<int>(std::upper_bound(times.data(), times.data() + n, t0) - times.data()) - 1;
  k = std::max(k, 0);
  while (k < n) {
    const double seg_start = std::max(times[k], t0);
    double seg_end = (k + 1 < n) ? times[k + 1] : t1;
    seg_end = std::min(seg_end, t1);
    if (seg_end > seg_start) {
      out.emplace_back(k, seg_end - seg_start);
    }
    if (seg_end >= t1) {
      break;
    }
    ++k;
  }
  return out;
}

}  // namespace insight_vio

#include <insight_vio/triangulation.hpp>

#include <Eigen/Cholesky>

#include <cmath>
#include <limits>

namespace insight_vio {

Eigen::Vector3d triangulate_linear(const std::vector<Eigen::Matrix3d>& R_wc,
                                   const std::vector<Eigen::Vector3d>& p_wc,
                                   const Eigen::MatrixX2d& obs) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  const int n = static_cast<int>(obs.rows());
  for (int i = 0; i < n; ++i) {
    Eigen::Vector3d d = R_wc[i] * Eigen::Vector3d(obs(i, 0), obs(i, 1), 1.0);
    d.normalize();
    const Eigen::Matrix3d M = Eigen::Matrix3d::Identity() - d * d.transpose();
    A += M;
    b += M * p_wc[i];
  }
  return (A + 1e-12 * Eigen::Matrix3d::Identity()).ldlt().solve(b);
}

TriangulationResult triangulate_gn(const std::vector<Eigen::Matrix3d>& R_wc,
                                   const std::vector<Eigen::Vector3d>& p_wc,
                                   const Eigen::MatrixX2d& obs, int max_iters, double min_depth,
                                   double max_depth, double max_rms) {
  const int n = static_cast<int>(obs.rows());
  TriangulationResult fail;
  fail.ok = false;
  fail.rms_error = std::numeric_limits<double>::infinity();
  fail.iterations = 0;
  if (n < 2) {
    return fail;
  }

  const Eigen::Matrix3d& R_wa = R_wc[0];
  const Eigen::Vector3d& p_wa = p_wc[0];
  std::vector<Eigen::Matrix3d> Rs(n);
  std::vector<Eigen::Vector3d> ts(n);
  for (int i = 0; i < n; ++i) {
    Rs[i] = R_wc[i].transpose() * R_wa;
    ts[i] = R_wc[i].transpose() * (p_wa - p_wc[i]);
  }

  Eigen::Vector3d p_lin = triangulate_linear(R_wc, p_wc, obs);
  Eigen::Vector3d p_a = R_wa.transpose() * (p_lin - p_wa);
  if (p_a[2] < min_depth) {
    p_a = Eigen::Vector3d(obs(0, 0), obs(0, 1), 1.0) * 3.0;
  }
  Eigen::Vector3d x(p_a[0] / p_a[2], p_a[1] / p_a[2], 1.0 / p_a[2]);

  auto residuals = [&](const Eigen::Vector3d& xx, Eigen::VectorXd& r, Eigen::MatrixX3d& J) {
    r.resize(2 * n);
    J.resize(2 * n, 3);
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d h =
          Rs[i] * Eigen::Vector3d(xx[0], xx[1], 1.0) + xx[2] * ts[i];
      const double hz = (std::abs(h[2]) > 1e-9) ? h[2] : 1e-9;
      r.segment<2>(2 * i) = obs.row(i).transpose() - h.head<2>() / hz;
      Eigen::Matrix<double, 2, 3> dz_dh;
      dz_dh << 1.0 / hz, 0.0, -h[0] / (hz * hz), 0.0, 1.0 / hz, -h[1] / (hz * hz);
      Eigen::Matrix3d dh_dx;
      dh_dx.col(0) = Rs[i].col(0);
      dh_dx.col(1) = Rs[i].col(1);
      dh_dx.col(2) = ts[i];
      J.block<2, 3>(2 * i, 0) = dz_dh * dh_dx;
    }
  };

  double lam = 1e-3;
  Eigen::VectorXd r;
  Eigen::MatrixX3d J;
  residuals(x, r, J);
  double cost = r.squaredNorm();
  int it = 0;
  for (it = 1; it <= max_iters; ++it) {
    const Eigen::Matrix3d H = J.transpose() * J;
    const Eigen::Vector3d g = J.transpose() * r;
    Eigen::Matrix3d lhs = H;
    lhs.diagonal().array() += lam * (H.diagonal().array() + 1e-12);
    const Eigen::Vector3d dx = lhs.ldlt().solve(g);
    const Eigen::Vector3d x_new = x + dx;
    Eigen::VectorXd r_new;
    Eigen::MatrixX3d J_new;
    residuals(x_new, r_new, J_new);
    const double cost_new = r_new.squaredNorm();
    if (cost_new < cost) {
      x = x_new;
      r = r_new;
      J = J_new;
      cost = cost_new;
      lam = std::max(lam * 0.1, 1e-10);
      if (dx.norm() < 1e-9) {
        break;
      }
    } else {
      lam *= 10.0;
      if (lam > 1e8) {
        break;
      }
    }
  }

  if (x[2] <= 0.0) {
    return fail;
  }
  p_a = Eigen::Vector3d(x[0], x[1], 1.0) / x[2];
  const Eigen::Vector3d p_w = R_wa * p_a + p_wa;
  for (int i = 0; i < n; ++i) {
    const double d = (R_wc[i].transpose() * (p_w - p_wc[i]))[2];
    if (d < min_depth || d > max_depth) {
      TriangulationResult bad;
      bad.ok = false;
      bad.p_w = p_w;
      bad.anchor_inv_depth = x;
      bad.rms_error = std::sqrt(cost / n);
      bad.iterations = it;
      return bad;
    }
  }
  const double rms = std::sqrt(cost / n);
  TriangulationResult ok;
  ok.ok = rms < max_rms;
  ok.p_w = p_w;
  ok.anchor_inv_depth = x;
  ok.rms_error = rms;
  ok.iterations = it;
  return ok;
}

}  // namespace insight_vio

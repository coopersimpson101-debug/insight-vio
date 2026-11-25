#pragma once
// Multi-view feature triangulation. Matches insight_vio/triangulation.py.

#include <Eigen/Dense>

#include <vector>

namespace insight_vio {

struct TriangulationResult {
  bool ok = false;
  Eigen::Vector3d p_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_inv_depth = Eigen::Vector3d::Zero();  // (alpha, beta, rho)
  double rms_error = 0.0;
  int iterations = 0;
};

Eigen::Vector3d triangulate_linear(const std::vector<Eigen::Matrix3d>& R_wc,
                                   const std::vector<Eigen::Vector3d>& p_wc,
                                   const Eigen::MatrixX2d& obs);

TriangulationResult triangulate_gn(const std::vector<Eigen::Matrix3d>& R_wc,
                                   const std::vector<Eigen::Vector3d>& p_wc,
                                   const Eigen::MatrixX2d& obs, int max_iters = 15,
                                   double min_depth = 0.1, double max_depth = 60.0,
                                   double max_rms = 0.02);

}  // namespace insight_vio

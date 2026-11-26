#pragma once

#include <Eigen/Dense>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace insight_vio {
namespace test {

inline std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) {
    return b;
  }
  if (a.back() == '/' || a.back() == '\\') {
    return a + b;
  }
  return a + "/" + b;
}

inline Eigen::MatrixXd load_matrix(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open golden file: " + path);
  }
  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("empty golden file: " + path);
  }
  std::istringstream hs(line);
  std::vector<int> shape;
  int s = 0;
  while (hs >> s) {
    shape.push_back(s);
  }
  if (shape.empty() || shape.size() > 2) {
    throw std::runtime_error("bad shape header in " + path);
  }
  const int rows = shape[0];
  const int cols = (shape.size() == 1) ? 1 : shape[1];
  Eigen::MatrixXd m(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (!(in >> m(r, c))) {
        throw std::runtime_error("truncated golden matrix in " + path);
      }
    }
  }
  return m;
}

inline Eigen::VectorXd load_vector(const std::string& path) {
  const Eigen::MatrixXd m = load_matrix(path);
  if (m.cols() == 1) {
    return m.col(0);
  }
  if (m.rows() == 1) {
    return m.row(0).transpose();
  }
  throw std::runtime_error("expected vector golden in " + path);
}

}  // namespace test
}  // namespace insight_vio

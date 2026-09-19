#include "internal.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace msm3d {

cv::Vec3d rotateAroundAxis(const cv::Vec3d& vector, const cv::Vec3d& axis,
                           double angle_rad) {
  cv::Mat rotation;
  cv::Rodrigues(axis * angle_rad, rotation);

  const cv::Mat input =
      (cv::Mat_<double>(3, 1) << vector[0], vector[1], vector[2]);
  const cv::Mat output = rotation * input;

  return cv::Vec3d(output.at<double>(0), output.at<double>(1),
                   output.at<double>(2));
}

double invertAngleModel(const RationalAngleModel& model, double target_theta,
                        double min_psi, double max_psi) {
  if (!model.valid || !(max_psi > min_psi)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double lo = min_psi;
  double hi = max_psi;
  const double theta_lo = model.evaluate(lo);
  const double theta_hi = model.evaluate(hi);
  const bool increasing = theta_hi >= theta_lo;

  const double theta_min = std::min(theta_lo, theta_hi);
  const double theta_max = std::max(theta_lo, theta_hi);
  if (target_theta < theta_min - 1e-10 || target_theta > theta_max + 1e-10) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  for (int iter = 0; iter < 80; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double theta_mid = model.evaluate(mid);

    if (increasing) {
      if (theta_mid < target_theta) {
        lo = mid;
      } else {
        hi = mid;
      }
    } else {
      if (theta_mid > target_theta) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
  }

  return 0.5 * (lo + hi);
}

std::vector<double> makeUniformSamples(double min_value, double max_value,
                                       int count) {
  std::vector<double> samples;
  if (count <= 0) {
    return samples;
  }

  samples.reserve(static_cast<std::size_t>(count));
  if (count == 1) {
    samples.push_back(0.5 * (min_value + max_value));
    return samples;
  }

  for (int i = 0; i < count; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(count - 1);
    samples.push_back(min_value + t * (max_value - min_value));
  }

  return samples;
}

double computeAngleModelRmse(const std::vector<DiscretePlane>& planes,
                             const std::vector<double>& thetas,
                             const RationalAngleModel& model,
                             double* out_max_abs_error) {
  double weighted_sum_sq = 0.0;
  double weight_sum = 0.0;
  double max_abs_error = 0.0;

  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (!planes[i].valid) {
      continue;
    }

    const double prediction = model.evaluate(planes[i].psi);
    const double residual = thetas[i] - prediction;
    const double weight = std::max(planes[i].confidence, 1e-6);

    weighted_sum_sq += weight * residual * residual;
    weight_sum += weight;
    max_abs_error = std::max(max_abs_error, std::abs(residual));
  }

  if (out_max_abs_error != nullptr) {
    *out_max_abs_error = max_abs_error;
  }

  return (weight_sum > 0.0) ? std::sqrt(weighted_sum_sq / weight_sum) : 0.0;
}
bool solveCenterFromPlanes(const std::vector<DiscretePlane>& planes,
                           const cv::Vec3d& u, const cv::Vec3d& v,
                           cv::Vec3d& out_center) {
  std::vector<const DiscretePlane*> valid_planes;
  for (const auto& plane : planes) {
    if (plane.valid) {
      valid_planes.push_back(&plane);
    }
  }

  if (valid_planes.size() < 3) {
    return false;
  }

  cv::Mat A(static_cast<int>(valid_planes.size()), 2, CV_64F);
  cv::Mat b(static_cast<int>(valid_planes.size()), 1, CV_64F);

  for (std::size_t i = 0; i < valid_planes.size(); ++i) {
    const auto& plane = *valid_planes[i];
    const double sqrt_weight = std::sqrt(std::max(plane.confidence, 1e-8));

    A.at<double>(static_cast<int>(i), 0) = sqrt_weight * plane.normal.dot(u);
    A.at<double>(static_cast<int>(i), 1) = sqrt_weight * plane.normal.dot(v);
    b.at<double>(static_cast<int>(i), 0) = -sqrt_weight * plane.d;
  }

  cv::Mat parameters;
  if (!cv::solve(A, b, parameters, cv::DECOMP_SVD)) {
    return false;
  }

  out_center = parameters.at<double>(0) * u + parameters.at<double>(1) * v;
  return true;
}
bool solveNominalRotationGeometry(const std::vector<DiscretePlane>& planes,
                                  double ref_psi, cv::Vec3d& out_w,
                                  cv::Vec3d& out_S0, cv::Vec3d& out_n0,
                                  cv::Vec3d& out_u, cv::Vec3d& out_v) {
  std::vector<const DiscretePlane*> valid_planes;
  valid_planes.reserve(planes.size());

  for (const auto& plane : planes) {
    if (plane.valid) {
      valid_planes.push_back(&plane);
    }
  }

  if (valid_planes.size() < 3) {
    return false;
  }

  cv::Mat normal_cov = cv::Mat::zeros(3, 3, CV_64F);
  for (const auto* plane : valid_planes) {
    const cv::Mat n = (cv::Mat_<double>(3, 1) << plane->normal[0],
                       plane->normal[1], plane->normal[2]);
    normal_cov += plane->confidence * (n * n.t());
  }

  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
  cv::eigen(normal_cov, eigenvalues, eigenvectors);

  out_w =
      cv::Vec3d(eigenvectors.at<double>(2, 0), eigenvectors.at<double>(2, 1),
                eigenvectors.at<double>(2, 2));
  out_w = cv::normalize(out_w);

  const DiscretePlane* first_plane = valid_planes.front();
  const DiscretePlane* last_plane = valid_planes.back();

  cv::Vec3d first_normal =
      first_plane->normal - first_plane->normal.dot(out_w) * out_w;
  cv::Vec3d last_normal =
      last_plane->normal - last_plane->normal.dot(out_w) * out_w;

  if (cv::norm(first_normal) > kEpsilon && cv::norm(last_normal) > kEpsilon) {
    first_normal = cv::normalize(first_normal);
    last_normal = cv::normalize(last_normal);
    const double signed_turn = first_normal.cross(last_normal).dot(out_w);

    if (signed_turn < 0.0) {
      out_w = -out_w;
    } else if (std::abs(signed_turn) < 1e-10 && out_w[1] < 0.0) {
      out_w = -out_w;
    }
  } else if (out_w[1] < 0.0) {
    out_w = -out_w;
  }

  const cv::Vec3d reference_axis =
      (std::abs(out_w[0]) > 0.9) ? cv::Vec3d(0, 1, 0) : cv::Vec3d(1, 0, 0);
  out_u = cv::normalize(out_w.cross(reference_axis));
  out_v = cv::normalize(out_w.cross(out_u));

  cv::Mat A_s0(static_cast<int>(valid_planes.size()), 2, CV_64F);
  cv::Mat b_s0(static_cast<int>(valid_planes.size()), 1, CV_64F);

  for (std::size_t i = 0; i < valid_planes.size(); ++i) {
    const auto& plane = *valid_planes[i];
    const double sqrt_weight = std::sqrt(std::max(plane.confidence, 0.0));

    A_s0.at<double>(static_cast<int>(i), 0) =
        sqrt_weight * plane.normal.dot(out_u);
    A_s0.at<double>(static_cast<int>(i), 1) =
        sqrt_weight * plane.normal.dot(out_v);
    b_s0.at<double>(static_cast<int>(i), 0) = -sqrt_weight * plane.d;
  }

  cv::Mat center_parameters;
  if (!cv::solve(A_s0, b_s0, center_parameters, cv::DECOMP_SVD)) {
    return false;
  }

  out_S0 = center_parameters.at<double>(0) * out_u +
           center_parameters.at<double>(1) * out_v;

  const DiscretePlane* left_plane = nullptr;
  const DiscretePlane* right_plane = nullptr;

  for (const auto* plane : valid_planes) {
    if (plane->psi <= ref_psi) {
      left_plane = plane;
    }
    if (plane->psi >= ref_psi) {
      right_plane = plane;
      break;
    }
  }

  if (left_plane == nullptr) {
    left_plane = valid_planes.front();
  }
  if (right_plane == nullptr) {
    right_plane = valid_planes.back();
  }

  cv::Vec3d left_normal =
      left_plane->normal - left_plane->normal.dot(out_w) * out_w;
  cv::Vec3d right_normal =
      right_plane->normal - right_plane->normal.dot(out_w) * out_w;

  if (cv::norm(left_normal) < kEpsilon || cv::norm(right_normal) < kEpsilon) {
    return false;
  }

  left_normal = cv::normalize(left_normal);
  right_normal = cv::normalize(right_normal);

  if (left_plane == right_plane ||
      std::abs(right_plane->psi - left_plane->psi) < 1e-12) {
    out_n0 = left_normal;
  } else {
    const double interpolation = std::clamp(
        (ref_psi - left_plane->psi) / (right_plane->psi - left_plane->psi), 0.0,
        1.0);

    const double cos_delta =
        std::clamp(left_normal.dot(right_normal), -1.0, 1.0);
    const double sin_delta = left_normal.cross(right_normal).dot(out_w);
    const double delta_angle = std::atan2(sin_delta, cos_delta);

    out_n0 = rotateAroundAxis(left_normal, out_w, interpolation * delta_angle);
    out_n0 = cv::normalize(out_n0 - out_n0.dot(out_w) * out_w);
  }

  return true;
}
std::vector<double> computeRelativeAngles(
    const std::vector<DiscretePlane>& planes, const cv::Vec3d& w,
    const cv::Vec3d& n0, bool enforce_monotonic) {
  std::vector<double> thetas(planes.size(), 0.0);
  std::vector<int> valid_indices;
  valid_indices.reserve(planes.size());

  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (!planes[i].valid) {
      continue;
    }

    const cv::Vec3d projected = planes[i].normal - planes[i].normal.dot(w) * w;
    if (cv::norm(projected) < kEpsilon) {
      continue;
    }

    const cv::Vec3d normal = cv::normalize(projected);
    const double cos_theta = std::clamp(n0.dot(normal), -1.0, 1.0);
    const double sin_theta = n0.cross(normal).dot(w);

    thetas[i] = std::atan2(sin_theta, cos_theta);
    valid_indices.push_back(static_cast<int>(i));
  }

  if (!enforce_monotonic || valid_indices.size() < 2) {
    return thetas;
  }

  struct Block {
    int begin = 0;
    int end = 0;
    double weight = 0.0;
    double mean = 0.0;
  };

  std::vector<Block> blocks;
  blocks.reserve(valid_indices.size());

  for (std::size_t k = 0; k < valid_indices.size(); ++k) {
    const int idx = valid_indices[k];
    const double weight =
        std::max(planes[static_cast<std::size_t>(idx)].confidence, 1e-6);

    blocks.push_back({static_cast<int>(k), static_cast<int>(k), weight,
                      thetas[static_cast<std::size_t>(idx)]});

    while (blocks.size() >= 2) {
      const std::size_t n = blocks.size();
      if (blocks[n - 2].mean <= blocks[n - 1].mean) {
        break;
      }

      const Block right = blocks.back();
      blocks.pop_back();
      Block& left = blocks.back();

      const double merged_weight = left.weight + right.weight;
      left.mean =
          (left.mean * left.weight + right.mean * right.weight) / merged_weight;
      left.weight = merged_weight;
      left.end = right.end;
    }
  }

  for (const auto& block : blocks) {
    for (int k = block.begin; k <= block.end; ++k) {
      const int idx = valid_indices[static_cast<std::size_t>(k)];
      thetas[static_cast<std::size_t>(idx)] = block.mean;
    }
  }

  return thetas;
}
RationalAngleModel fitRationalAngleModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    double psi_ref) {
  RationalAngleModel model;
  model.psi_ref = psi_ref;
  model.valid = false;

  std::vector<int> valid_indices;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (planes[i].valid) {
      valid_indices.push_back(static_cast<int>(i));
    }
  }

  if (valid_indices.size() < 3) {
    return model;
  }

  cv::Mat A(static_cast<int>(valid_indices.size()), 2, CV_64F);
  cv::Mat b(static_cast<int>(valid_indices.size()), 1, CV_64F);

  for (std::size_t row = 0; row < valid_indices.size(); ++row) {
    const int idx = valid_indices[row];
    const auto& plane = planes[static_cast<std::size_t>(idx)];
    const double theta = thetas[static_cast<std::size_t>(idx)];
    const double delta_psi = plane.psi - psi_ref;
    const double tan_theta = std::tan(theta);
    const double cos_theta = std::cos(theta);

    const double total_weight =
        std::max(plane.confidence, 1e-6) * cos_theta * cos_theta;
    const double sqrt_weight = std::sqrt(total_weight);

    A.at<double>(static_cast<int>(row), 0) = sqrt_weight * tan_theta;
    A.at<double>(static_cast<int>(row), 1) =
        sqrt_weight * tan_theta * delta_psi;
    b.at<double>(static_cast<int>(row), 0) = sqrt_weight * delta_psi;
  }

  cv::Mat parameters;
  if (!cv::solve(A, b, parameters, cv::DECOMP_SVD)) {
    return model;
  }

  model.b0 = parameters.at<double>(0);
  model.b1 = parameters.at<double>(1);

  if (!std::isfinite(model.b0) || !std::isfinite(model.b1) ||
      std::abs(model.b0) < 1e-12) {
    return model;
  }

  model.valid = true;
  return model;
}
HarmonicDriftModel fitHarmonicDriftModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    const cv::Vec3d& w, const cv::Vec3d& initial_s0, const cv::Vec3d& u,
    const cv::Vec3d& v, cv::Vec3d& out_reference_s0, int order,
    double regularization) {
  (void)w;

  HarmonicDriftModel model;
  model.order = order;
  model.valid = false;
  out_reference_s0 = initial_s0;

  if (order <= 0) {
    return model;
  }

  std::vector<int> valid_indices;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (planes[i].valid) {
      valid_indices.push_back(static_cast<int>(i));
    }
  }

  // Two reference-center corrections plus four coefficients per harmonic:
  // u*(cos(k*theta)-1), u*sin(k*theta),
  // v*(cos(k*theta)-1), v*sin(k*theta).
  const int num_harmonic_vars = 4 * order;
  const int num_vars = 2 + num_harmonic_vars;

  if (static_cast<int>(valid_indices.size()) < num_vars + 2) {
    return model;
  }

  const int num_data_rows = static_cast<int>(valid_indices.size());
  const int num_regularization_rows = num_harmonic_vars;
  cv::Mat A =
      cv::Mat::zeros(num_data_rows + num_regularization_rows, num_vars, CV_64F);
  cv::Mat b =
      cv::Mat::zeros(num_data_rows + num_regularization_rows, 1, CV_64F);

  for (int row = 0; row < num_data_rows; ++row) {
    const int idx = valid_indices[static_cast<std::size_t>(row)];
    const auto& plane = planes[static_cast<std::size_t>(idx)];
    const double theta = thetas[static_cast<std::size_t>(idx)];
    const auto& n = plane.normal;

    const double nu = n.dot(u);
    const double nv = n.dot(v);
    const double sqrt_weight = std::sqrt(std::max(plane.confidence, 1e-8));

    // Reference-center correction relative to initial_s0.
    A.at<double>(row, 0) = sqrt_weight * nu;
    A.at<double>(row, 1) = sqrt_weight * nv;

    for (int k = 1; k <= order; ++k) {
      const double angle = static_cast<double>(k) * theta;
      const double cos_basis = std::cos(angle) - 1.0;
      const double sin_basis = std::sin(angle);
      const int col = 2 + 4 * (k - 1);

      A.at<double>(row, col + 0) = sqrt_weight * nu * cos_basis;
      A.at<double>(row, col + 1) = sqrt_weight * nu * sin_basis;
      A.at<double>(row, col + 2) = sqrt_weight * nv * cos_basis;
      A.at<double>(row, col + 3) = sqrt_weight * nv * sin_basis;
    }

    b.at<double>(row, 0) = -sqrt_weight * (n.dot(initial_s0) + plane.d);
  }

  // Solve the regularized least-squares problem directly with SVD instead of
  // forming normal equations. Higher harmonics receive slightly stronger
  // regularization.
  const double lambda = std::max(regularization, 0.0);
  int reg_row = num_data_rows;

  for (int k = 1; k <= order; ++k) {
    const double sqrt_lambda = std::sqrt(lambda) * static_cast<double>(k);
    const int col = 2 + 4 * (k - 1);

    for (int j = 0; j < 4; ++j) {
      A.at<double>(reg_row, col + j) = sqrt_lambda;
      ++reg_row;
    }
  }

  cv::Mat parameters;
  if (!cv::solve(A, b, parameters, cv::DECOMP_SVD)) {
    return model;
  }

  const double center_u = parameters.at<double>(0);
  const double center_v = parameters.at<double>(1);
  out_reference_s0 = initial_s0 + center_u * u + center_v * v;

  model.beta_u.resize(2 * order, 0.0);
  model.beta_v.resize(2 * order, 0.0);

  for (int k = 1; k <= order; ++k) {
    const int col = 2 + 4 * (k - 1);
    const int dst_idx = 2 * (k - 1);

    model.beta_u[dst_idx + 0] = parameters.at<double>(col + 0);
    model.beta_u[dst_idx + 1] = parameters.at<double>(col + 1);
    model.beta_v[dst_idx + 0] = parameters.at<double>(col + 2);
    model.beta_v[dst_idx + 1] = parameters.at<double>(col + 3);
  }

  model.valid = true;
  return model;
}

}  // namespace msm3d

#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace msm3d {
namespace {

double computePlaneConfidence(double rms_mm, double thickness_ratio,
                              const MsmCalibrationOptions& options) {
  if (!std::isfinite(rms_mm) || rms_mm < 0.0 ||
      !std::isfinite(thickness_ratio) || thickness_ratio < 0.0) {
    return 0.0;
  }

  const double rms_floor =
      std::max(options.plane_confidence_rms_floor_mm, 1e-6);
  const double rms_weight = 1.0 / (rms_mm * rms_mm + rms_floor * rms_floor);

  const double thickness_scale = std::max(options.thickness_soft_scale, 1e-6);
  const double normalized_thickness = thickness_ratio / thickness_scale;

  // A gentle soft penalty. The broad max_thickness_ratio is still retained as
  // a sanity limit for clearly non-planar observations.
  const double thickness_weight =
      1.0 / std::sqrt(1.0 + normalized_thickness * normalized_thickness);

  return rms_weight * thickness_weight;
}

double weightedMedian(const std::vector<double>& values,
                      const std::vector<double>& weights) {
  if (values.empty() || values.size() != weights.size()) {
    return 0.0;
  }

  std::vector<std::pair<double, double>> pairs;
  pairs.reserve(values.size());

  double total_weight = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]) || !std::isfinite(weights[i]) ||
        weights[i] <= 0.0) {
      continue;
    }
    pairs.emplace_back(values[i], weights[i]);
    total_weight += weights[i];
  }

  if (pairs.empty() || total_weight <= 0.0) {
    return 0.0;
  }

  std::sort(pairs.begin(), pairs.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  const double half_weight = 0.5 * total_weight;
  double accumulated = 0.0;
  for (const auto& item : pairs) {
    accumulated += item.second;
    if (accumulated >= half_weight) {
      return item.first;
    }
  }

  return pairs.back().first;
}
bool computeWeightedPlaneSvd(const std::vector<cv::Vec3d>& points,
                             const std::vector<double>& weights,
                             cv::Vec3d& out_center, cv::Vec3d& out_normal,
                             cv::Vec3d& out_singular_values) {
  if (points.size() < 3 || points.size() != weights.size()) {
    return false;
  }

  double sum_weights = 0.0;
  cv::Vec3d center(0.0, 0.0, 0.0);

  for (std::size_t i = 0; i < points.size(); ++i) {
    const double w = weights[i];
    if (!std::isfinite(w) || w <= 0.0) {
      continue;
    }
    center += points[i] * w;
    sum_weights += w;
  }

  if (sum_weights <= kEpsilon) {
    return false;
  }

  center /= sum_weights;

  cv::Mat A(static_cast<int>(points.size()), 3, CV_64F, cv::Scalar(0.0));

  for (std::size_t i = 0; i < points.size(); ++i) {
    const double w = std::max(weights[i], 0.0);
    const double sqrt_w = std::sqrt(w);
    const cv::Vec3d q = points[i] - center;

    A.at<double>(static_cast<int>(i), 0) = sqrt_w * q[0];
    A.at<double>(static_cast<int>(i), 1) = sqrt_w * q[1];
    A.at<double>(static_cast<int>(i), 2) = sqrt_w * q[2];
  }

  cv::Mat singular_values;
  cv::Mat u;
  cv::Mat vt;
  cv::SVD::compute(A, singular_values, u, vt);

  if (singular_values.total() < 3 || vt.rows < 3 || vt.cols < 3) {
    return false;
  }

  out_center = center;
  out_normal =
      cv::Vec3d(vt.at<double>(2, 0), vt.at<double>(2, 1), vt.at<double>(2, 2));
  out_normal = cv::normalize(out_normal);

  out_singular_values =
      cv::Vec3d(singular_values.at<double>(0), singular_values.at<double>(1),
                singular_values.at<double>(2));
  return true;
}

}  // namespace

void normalizePlaneConfidences(std::vector<DiscretePlane>& planes) {
  std::vector<double> confidences;
  confidences.reserve(planes.size());

  for (const auto& plane : planes) {
    if (plane.valid && std::isfinite(plane.confidence) &&
        plane.confidence > 0.0) {
      confidences.push_back(plane.confidence);
    }
  }

  if (confidences.empty()) {
    return;
  }

  const std::size_t mid = confidences.size() / 2;
  std::nth_element(confidences.begin(),
                   confidences.begin() + static_cast<std::ptrdiff_t>(mid),
                   confidences.end());
  const double median_conf = std::max(confidences[mid], kEpsilon);

  for (auto& plane : planes) {
    if (!plane.valid) {
      plane.confidence = 0.0;
      continue;
    }

    const double normalized =
        std::clamp(plane.confidence / median_conf, 0.1, 10.0);
    plane.confidence = normalized;
  }
}
DiscretePlane fitOffsetForFixedNormal(const PosePointGroups& points_by_pose,
                                      double psi, const cv::Vec3d& fixed_normal,
                                      const DiscretePlane& geometry_source,
                                      const MsmCalibrationOptions& options) {
  DiscretePlane plane = geometry_source;
  plane.psi = psi;
  plane.normal = cv::normalize(fixed_normal);
  plane.valid = false;

  std::vector<cv::Vec3d> points;
  std::vector<double> base_weights;
  std::vector<double> projected_offsets;

  int contributing_poses = 0;
  std::size_t total_points = 0;
  for (const auto& pose_points : points_by_pose) {
    if (!pose_points.empty()) {
      ++contributing_poses;
      total_points += pose_points.size();
    }
  }

  plane.pose_count = contributing_poses;
  plane.point_count = static_cast<int>(total_points);

  if (contributing_poses <= 0 || total_points < 30) {
    plane.confidence = 0.0;
    return plane;
  }

  points.reserve(total_points);
  base_weights.reserve(total_points);
  projected_offsets.reserve(total_points);

  for (const auto& pose_points : points_by_pose) {
    if (pose_points.empty()) {
      continue;
    }

    const double point_weight = 1.0 / static_cast<double>(pose_points.size());

    for (const auto& point : pose_points) {
      points.push_back(point);
      base_weights.push_back(point_weight);
      projected_offsets.push_back(-plane.normal.dot(point));
    }
  }

  double d = weightedMedian(projected_offsets, base_weights);

  constexpr int kMaxIterations = 8;
  constexpr double kHuber = 1.345;
  std::vector<double> combined_weights(base_weights.size(), 0.0);

  for (int iter = 0; iter < kMaxIterations; ++iter) {
    std::vector<double> abs_residuals(points.size(), 0.0);

    for (std::size_t i = 0; i < points.size(); ++i) {
      abs_residuals[i] = std::abs(plane.normal.dot(points[i]) + d);
    }

    const double mad = weightedMedian(abs_residuals, base_weights);
    const double sigma = std::max(1.4826 * mad, 0.02);

    double weighted_sum = 0.0;
    double weight_sum = 0.0;

    for (std::size_t i = 0; i < points.size(); ++i) {
      const double normalized = abs_residuals[i] / sigma;
      const double robust_weight =
          (normalized > kHuber) ? (kHuber / normalized) : 1.0;

      combined_weights[i] = base_weights[i] * robust_weight;
      weighted_sum += combined_weights[i] * projected_offsets[i];
      weight_sum += combined_weights[i];
    }

    if (weight_sum <= kEpsilon) {
      break;
    }

    const double next_d = weighted_sum / weight_sum;
    if (std::abs(next_d - d) < 1e-10) {
      d = next_d;
      break;
    }
    d = next_d;
  }

  plane.d = d;

  double weighted_sum_sq = 0.0;
  double weighted_inlier_mass = 0.0;
  double total_base_mass = 0.0;
  int inlier_count = 0;

  for (std::size_t i = 0; i < points.size(); ++i) {
    const double residual = plane.normal.dot(points[i]) + plane.d;
    const double base_weight = base_weights[i];
    total_base_mass += base_weight;

    if (std::abs(residual) <= options.plane_inlier_threshold_mm) {
      weighted_sum_sq += base_weight * residual * residual;
      weighted_inlier_mass += base_weight;
      ++inlier_count;
    }
  }

  plane.inlier_count = inlier_count;
  plane.inlier_ratio = (total_base_mass > kEpsilon)
                           ? (weighted_inlier_mass / total_base_mass)
                           : 0.0;

  plane.rms_mm = (weighted_inlier_mass > kEpsilon)
                     ? std::sqrt(weighted_sum_sq / weighted_inlier_mass)
                     : std::numeric_limits<double>::infinity();

  const bool pose_ok = plane.pose_count >= options.min_covisible_poses;
  const bool geometry_ok =
      geometry_source.valid &&
      geometry_source.spread_ratio >= options.min_spread_ratio &&
      geometry_source.thickness_ratio <= options.max_thickness_ratio;
  const bool rms_ok =
      std::isfinite(plane.rms_mm) && plane.rms_mm <= options.max_plane_rms_mm;

  plane.valid = pose_ok && geometry_ok && rms_ok && plane.inlier_count >= 30;

  if (plane.valid) {
    plane.confidence =
        computePlaneConfidence(plane.rms_mm, plane.thickness_ratio, options);
  } else {
    plane.confidence = 0.0;
  }

  return plane;
}
DiscretePlane fitPlaneRobustTLS(
    const std::vector<std::vector<cv::Vec3d>>& points_by_pose, double psi,
    const MsmCalibrationOptions& options) {
  DiscretePlane plane;
  plane.psi = psi;
  plane.valid = false;

  std::vector<cv::Vec3d> points;
  std::vector<double> base_weights;

  int contributing_poses = 0;
  std::size_t total_points = 0;
  for (const auto& pose_points : points_by_pose) {
    if (!pose_points.empty()) {
      ++contributing_poses;
      total_points += pose_points.size();
    }
  }

  plane.pose_count = contributing_poses;
  plane.point_count = static_cast<int>(total_points);

  if (contributing_poses <= 0 || total_points < 30) {
    return plane;
  }

  points.reserve(total_points);
  base_weights.reserve(total_points);

  // Equal total weight per pose:
  // sum_j w_ij = 1 for every contributing pose i.
  for (const auto& pose_points : points_by_pose) {
    if (pose_points.empty()) {
      continue;
    }

    const double point_weight = 1.0 / static_cast<double>(pose_points.size());
    for (const auto& point : pose_points) {
      points.push_back(point);
      base_weights.push_back(point_weight);
    }
  }

  cv::Vec3d center;
  cv::Vec3d normal;
  cv::Vec3d singular_values;
  if (!computeWeightedPlaneSvd(points, base_weights, center, normal,
                               singular_values)) {
    return plane;
  }

  double d = -normal.dot(center);

  constexpr int kMaxIrlsIterations = 5;
  constexpr double kHuber = 1.345;
  std::vector<double> combined_weights = base_weights;

  for (int iter = 0; iter < kMaxIrlsIterations; ++iter) {
    std::vector<double> abs_residuals(points.size(), 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
      abs_residuals[i] = std::abs(normal.dot(points[i]) + d);
    }

    const double mad = weightedMedian(abs_residuals, base_weights);
    const double sigma = std::max(1.4826 * mad, 0.05);

    for (std::size_t i = 0; i < points.size(); ++i) {
      const double normalized_residual = abs_residuals[i] / sigma;
      const double huber_weight =
          (normalized_residual > kHuber) ? (kHuber / normalized_residual) : 1.0;

      combined_weights[i] = base_weights[i] * huber_weight;
    }

    if (!computeWeightedPlaneSvd(points, combined_weights, center, normal,
                                 singular_values)) {
      return plane;
    }

    d = -normal.dot(center);
  }

  // Final robust weighted geometry diagnostics.
  if (!computeWeightedPlaneSvd(points, combined_weights, center, normal,
                               singular_values)) {
    return plane;
  }
  d = -normal.dot(center);

  const double s1 = singular_values[0];
  const double s2 = singular_values[1];
  const double s3 = singular_values[2];

  plane.spread_ratio = s2 / (s1 + kEpsilon);
  plane.thickness_ratio = s3 / (s2 + kEpsilon);

  double weighted_sum_sq = 0.0;
  double weighted_inlier_mass = 0.0;
  double total_base_mass = 0.0;
  int inlier_count = 0;

  for (std::size_t i = 0; i < points.size(); ++i) {
    const double residual = normal.dot(points[i]) + d;
    const double base_weight = base_weights[i];
    total_base_mass += base_weight;

    if (std::abs(residual) <= options.plane_inlier_threshold_mm) {
      weighted_sum_sq += base_weight * residual * residual;
      weighted_inlier_mass += base_weight;
      ++inlier_count;
    }
  }

  plane.normal = normal;
  plane.d = d;
  plane.inlier_count = inlier_count;

  if (weighted_inlier_mass > kEpsilon) {
    plane.rms_mm = std::sqrt(weighted_sum_sq / weighted_inlier_mass);
  } else {
    plane.rms_mm = std::numeric_limits<double>::infinity();
  }

  plane.inlier_ratio = (total_base_mass > kEpsilon)
                           ? (weighted_inlier_mass / total_base_mass)
                           : 0.0;

  const bool pose_ok = plane.pose_count >= options.min_covisible_poses;
  const bool spread_ok = plane.spread_ratio >= options.min_spread_ratio;
  const bool thickness_ok =
      plane.thickness_ratio <= options.max_thickness_ratio;
  const bool rms_ok =
      std::isfinite(plane.rms_mm) && plane.rms_mm <= options.max_plane_rms_mm;

  plane.valid = pose_ok && spread_ok && thickness_ok && rms_ok &&
                plane.inlier_count >= 30;

  if (plane.valid) {
    plane.confidence =
        computePlaneConfidence(plane.rms_mm, plane.thickness_ratio, options);
  } else {
    plane.confidence = 0.0;
  }

  return plane;
}

DiscretePlane fitPlaneRobustTLS(const std::vector<cv::Vec3d>& points,
                                double psi,
                                const MsmCalibrationOptions& options) {
  std::vector<std::vector<cv::Vec3d>> grouped_points;
  if (!points.empty()) {
    grouped_points.push_back(points);
  }

  // Preserve the old standalone behavior: the flat overload is treated
  // as a single group and should not be rejected merely because the main
  // calibration requires multiple co-visible poses.
  MsmCalibrationOptions compatible_options = options;
  compatible_options.min_covisible_poses = 1;
  return fitPlaneRobustTLS(grouped_points, psi, compatible_options);
}

void enforceNormalConsistency(std::vector<DiscretePlane>& planes) {
  if (planes.empty()) return;

  cv::Vec3d ref_normal;
  bool found = false;
  for (const auto& plane : planes) {
    if (plane.valid) {
      ref_normal = plane.normal;
      found = true;
      break;
    }
  }

  if (!found) return;

  for (auto& plane : planes) {
    if (plane.valid && plane.normal.dot(ref_normal) < 0.0) {
      plane.normal = -plane.normal;
      plane.d = -plane.d;
    }
  }
}

}  // namespace msm3d

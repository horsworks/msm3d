#include "msm3d/msm_calibration.hpp"

#include "msm3d/camera_calibration.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace msm3d {
namespace {

constexpr double kEpsilon = 1e-12;

struct LocalPhaseFit {
  bool valid = false;
  double x = 0.0;
  double slope = 0.0;
  double rmse = std::numeric_limits<double>::infinity();
};

using PosePointGroups = std::vector<std::vector<cv::Vec3d>>;
using PlaneObservationSets = std::vector<PosePointGroups>;

struct ReconstructionDiagnostics {
  double rmse_3d_mm = 0.0;
  double model_plane_rmse_mm = 0.0;
  double ray_plane_denom_min = 0.0;
  double ray_plane_denom_p05 = 0.0;
  double ray_plane_denom_median = 0.0;
  double amplification = 0.0;
  int sample_count = 0;

  std::vector<double> phase_bin_rmse_3d_mm;
  std::vector<double> phase_bin_plane_rmse_mm;
  std::vector<int> phase_bin_sample_count;
};

std::vector<double> makeUniformSamples(double min_value, double max_value,
                                       int count);

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

LocalPhaseFit fitLocalPhaseLine(const cv::Mat& phase_f64, int y, int crossing_x,
                                double target_psi, const cv::Mat& mask,
                                int half_window, double min_abs_gradient,
                                double max_abs_gradient) {
  LocalPhaseFit result;

  const int cols = phase_f64.cols;
  const bool has_mask = !mask.empty() && mask.size() == phase_f64.size();

  // Include both crossing endpoints and a symmetric neighborhood.
  const int x_begin = std::max(0, crossing_x - half_window);
  const int x_end = std::min(cols - 1, crossing_x + 1 + half_window);

  const double* row_ptr = phase_f64.ptr<double>(y);
  const uchar* mask_ptr = has_mask ? mask.ptr<uchar>(y) : nullptr;

  double sum_x = 0.0;
  double sum_p = 0.0;
  int count = 0;

  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    sum_x += static_cast<double>(x);
    sum_p += p;
    ++count;
  }

  if (count < 3) {
    return result;
  }

  const double mean_x = sum_x / static_cast<double>(count);
  const double mean_p = sum_p / static_cast<double>(count);

  double sxx = 0.0;
  double sxp = 0.0;

  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    const double dx = static_cast<double>(x) - mean_x;
    const double dp = p - mean_p;
    sxx += dx * dx;
    sxp += dx * dp;
  }

  if (sxx <= kEpsilon) {
    return result;
  }

  const double slope = sxp / sxx;
  const double abs_slope = std::abs(slope);
  if (!std::isfinite(slope) || abs_slope < min_abs_gradient ||
      abs_slope > max_abs_gradient) {
    return result;
  }

  const double intercept = mean_p - slope * mean_x;
  const double fitted_x = (target_psi - intercept) / slope;

  // The fitted crossing must remain near the local support interval.
  if (!std::isfinite(fitted_x) ||
      fitted_x < static_cast<double>(x_begin) - 0.5 ||
      fitted_x > static_cast<double>(x_end) + 0.5) {
    return result;
  }

  double sum_sq = 0.0;
  int residual_count = 0;
  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    const double predicted = slope * static_cast<double>(x) + intercept;
    const double residual = p - predicted;
    sum_sq += residual * residual;
    ++residual_count;
  }

  if (residual_count < 3) {
    return result;
  }

  result.valid = true;
  result.x = fitted_x;
  result.slope = slope;
  result.rmse = std::sqrt(sum_sq / static_cast<double>(residual_count));
  return result;
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

void printPlaneQualityDiagnostics(const std::vector<DiscretePlane>& planes,
                                  const MsmCalibrationOptions& options) {
  int valid_count = 0;
  int rejected_pose_count = 0;
  int rejected_spread = 0;
  int rejected_thickness = 0;
  int rejected_rms = 0;

  std::vector<double> valid_rms;
  std::vector<double> valid_spread;
  std::vector<double> valid_thickness;
  std::vector<double> valid_confidence;

  for (const auto& plane : planes) {
    if (plane.pose_count < options.min_covisible_poses) {
      ++rejected_pose_count;
    }
    if (plane.spread_ratio < options.min_spread_ratio) {
      ++rejected_spread;
    }
    if (plane.thickness_ratio > options.max_thickness_ratio) {
      ++rejected_thickness;
    }
    if (plane.rms_mm > options.max_plane_rms_mm) {
      ++rejected_rms;
    }

    if (!plane.valid) {
      continue;
    }

    ++valid_count;
    valid_rms.push_back(plane.rms_mm);
    valid_spread.push_back(plane.spread_ratio);
    valid_thickness.push_back(plane.thickness_ratio);
    valid_confidence.push_back(plane.confidence);
  }

  auto print_stats = [](const char* name, std::vector<double> values) {
    if (values.empty()) {
      std::cout << "  " << name << ": n/a" << std::endl;
      return;
    }

    std::sort(values.begin(), values.end());
    const double min_v = values.front();
    const double med_v = values[values.size() / 2];
    const double max_v = values.back();

    std::cout << "  " << name << ": min=" << min_v << ", median=" << med_v
              << ", max=" << max_v << std::endl;
  };

  std::cout << "\nPlane quality diagnostics" << std::endl;
  std::cout << "  valid planes: " << valid_count << " / " << planes.size()
            << std::endl;
  print_stats("RMS [mm]", valid_rms);
  print_stats("spread ratio sigma2/sigma1", valid_spread);
  print_stats("thickness ratio sigma3/sigma2", valid_thickness);
  print_stats("normalized confidence", valid_confidence);

  std::cout << "  rejection counters (a plane may hit multiple rules):"
            << std::endl;
  std::cout << "    insufficient poses: " << rejected_pose_count << std::endl;
  std::cout << "    insufficient spread: " << rejected_spread << std::endl;
  std::cout << "    excessive thickness: " << rejected_thickness << std::endl;
  std::cout << "    excessive RMS: " << rejected_rms << std::endl;
}

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
                             double* out_max_abs_error = nullptr) {
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

void printPosePlaneResidualDiagnostics(const char* title,
                                       const std::vector<DiscretePlane>& planes,
                                       const PlaneObservationSets& observations,
                                       const std::vector<int>& pose_ids) {
  if (planes.size() != observations.size() || pose_ids.empty()) {
    return;
  }

  std::vector<double> sum_plane_rms_sq(pose_ids.size(), 0.0);
  std::vector<double> sum_plane_bias(pose_ids.size(), 0.0);
  std::vector<int> valid_plane_counts(pose_ids.size(), 0);

  for (std::size_t p = 0; p < planes.size(); ++p) {
    if (!planes[p].valid) {
      continue;
    }

    const auto& groups = observations[p];
    for (std::size_t pose_idx = 0;
         pose_idx < pose_ids.size() && pose_idx < groups.size(); ++pose_idx) {
      const auto& points = groups[pose_idx];
      if (points.empty()) {
        continue;
      }

      double sum_sq = 0.0;
      double sum_signed = 0.0;

      for (const auto& point : points) {
        const double residual = planes[p].normal.dot(point) + planes[p].d;
        sum_sq += residual * residual;
        sum_signed += residual;
      }

      const double count = static_cast<double>(points.size());
      const double group_rms = std::sqrt(sum_sq / count);
      const double group_bias = sum_signed / count;

      sum_plane_rms_sq[pose_idx] += group_rms * group_rms;
      sum_plane_bias[pose_idx] += group_bias;
      ++valid_plane_counts[pose_idx];
    }
  }

  std::cout << "\n" << title << std::endl;
  for (std::size_t i = 0; i < pose_ids.size(); ++i) {
    if (valid_plane_counts[i] <= 0) {
      std::cout << "  pose " << std::setw(2) << pose_ids[i] << ": n/a"
                << std::endl;
      continue;
    }

    const double plane_count = static_cast<double>(valid_plane_counts[i]);
    const double rms = std::sqrt(sum_plane_rms_sq[i] / plane_count);
    const double mean_bias = sum_plane_bias[i] / plane_count;

    std::cout << "  pose " << std::setw(2) << pose_ids[i]
              << ": plane RMSE=" << std::fixed << std::setprecision(5) << rms
              << " mm, mean signed bias=" << mean_bias
              << " mm, planes=" << valid_plane_counts[i] << std::endl;
  }
}

ReconstructionDiagnostics evaluateReconstructionDetailed(
    const MsmCalibrationResult& result, const cv::Mat& phase_map,
    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
    const BoardPlane& board_plane, double min_psi, double max_psi,
    int pixel_stride, int phase_bin_count = 0) {
  ReconstructionDiagnostics diagnostics;

  if (phase_map.empty() || pixel_stride <= 0) {
    return diagnostics;
  }

  cv::Mat phase64;
  if (phase_map.channels() > 1) {
    cv::extractChannel(phase_map, phase64, 0);
  } else {
    phase64 = phase_map;
  }
  phase64.convertTo(phase64, CV_64F);

  std::vector<cv::Point2d> pixels;
  std::vector<double> psis;

  for (int y = 0; y < phase64.rows; y += pixel_stride) {
    const double* row = phase64.ptr<double>(y);

    for (int x = 0; x < phase64.cols; x += pixel_stride) {
      const double psi = row[x];
      if (std::isfinite(psi) && psi >= min_psi && psi <= max_psi) {
        pixels.emplace_back(static_cast<double>(x), static_cast<double>(y));
        psis.push_back(psi);
      }
    }
  }

  if (pixels.empty()) {
    return diagnostics;
  }

  std::vector<cv::Point2d> normalized_points;
  cv::undistortPoints(pixels, normalized_points, camera_matrix, dist_coeffs);

  double sum_sq_3d = 0.0;
  double sum_sq_plane = 0.0;
  std::vector<double> denominators;
  denominators.reserve(normalized_points.size());

  const int bin_count = std::max(0, phase_bin_count);
  std::vector<double> bin_sum_sq_3d(static_cast<std::size_t>(bin_count), 0.0);
  std::vector<double> bin_sum_sq_plane(static_cast<std::size_t>(bin_count),
                                       0.0);
  std::vector<int> bin_samples(static_cast<std::size_t>(bin_count), 0);

  for (std::size_t i = 0; i < normalized_points.size(); ++i) {
    cv::Vec3d ray(normalized_points[i].x, normalized_points[i].y, 1.0);
    ray = cv::normalize(ray);

    const double board_denom = board_plane.normal.dot(ray);
    if (std::abs(board_denom) < 1e-6) {
      continue;
    }

    const double depth_gt = -board_plane.d / board_denom;
    if (!std::isfinite(depth_gt) || depth_gt < 120.0 || depth_gt > 250.0) {
      continue;
    }

    const cv::Vec3d X_gt = ray * depth_gt;

    cv::Vec3d model_normal;
    double model_d = 0.0;
    result.evaluatePlane(psis[i], model_normal, model_d);

    const double model_denom = model_normal.dot(ray);
    const double abs_model_denom = std::abs(model_denom);
    if (abs_model_denom < 1e-6) {
      continue;
    }

    const double plane_residual = model_normal.dot(X_gt) + model_d;

    // For a unit camera ray, the ray-wise 3D intersection error is exactly
    // |plane_residual| / |n dot ray|.
    const double error_3d = std::abs(plane_residual) / abs_model_denom;

    if (!std::isfinite(error_3d)) {
      continue;
    }

    sum_sq_3d += error_3d * error_3d;
    sum_sq_plane += plane_residual * plane_residual;
    denominators.push_back(abs_model_denom);

    if (bin_count > 0 && max_psi > min_psi) {
      const double normalized_phase = (psis[i] - min_psi) / (max_psi - min_psi);
      int bin = static_cast<int>(
          std::floor(normalized_phase * static_cast<double>(bin_count)));
      bin = std::clamp(bin, 0, bin_count - 1);

      bin_sum_sq_3d[static_cast<std::size_t>(bin)] += error_3d * error_3d;
      bin_sum_sq_plane[static_cast<std::size_t>(bin)] +=
          plane_residual * plane_residual;
      ++bin_samples[static_cast<std::size_t>(bin)];
    }
  }

  diagnostics.sample_count = static_cast<int>(denominators.size());

  if (diagnostics.sample_count <= 0) {
    return diagnostics;
  }

  const double count = static_cast<double>(diagnostics.sample_count);

  diagnostics.rmse_3d_mm = std::sqrt(sum_sq_3d / count);
  diagnostics.model_plane_rmse_mm = std::sqrt(sum_sq_plane / count);

  std::sort(denominators.begin(), denominators.end());
  diagnostics.ray_plane_denom_min = denominators.front();

  const std::size_t p05_index =
      std::min(denominators.size() - 1,
               static_cast<std::size_t>(
                   0.05 * static_cast<double>(denominators.size() - 1)));

  diagnostics.ray_plane_denom_p05 = denominators[p05_index];
  diagnostics.ray_plane_denom_median = denominators[denominators.size() / 2];

  if (diagnostics.model_plane_rmse_mm > 1e-12) {
    diagnostics.amplification =
        diagnostics.rmse_3d_mm / diagnostics.model_plane_rmse_mm;
  }

  diagnostics.phase_bin_rmse_3d_mm.assign(static_cast<std::size_t>(bin_count),
                                          0.0);
  diagnostics.phase_bin_plane_rmse_mm.assign(
      static_cast<std::size_t>(bin_count), 0.0);
  diagnostics.phase_bin_sample_count = bin_samples;

  for (int bin = 0; bin < bin_count; ++bin) {
    const int samples = bin_samples[static_cast<std::size_t>(bin)];
    if (samples <= 0) {
      continue;
    }

    diagnostics.phase_bin_rmse_3d_mm[static_cast<std::size_t>(bin)] =
        std::sqrt(bin_sum_sq_3d[static_cast<std::size_t>(bin)] /
                  static_cast<double>(samples));
    diagnostics.phase_bin_plane_rmse_mm[static_cast<std::size_t>(bin)] =
        std::sqrt(bin_sum_sq_plane[static_cast<std::size_t>(bin)] /
                  static_cast<double>(samples));
  }

  return diagnostics;
}

void printPhaseBinDiagnostics(const ReconstructionDiagnostics& diagnostics,
                              double min_psi, double max_psi) {
  const std::size_t bin_count = diagnostics.phase_bin_sample_count.size();
  if (bin_count == 0 || !(max_psi > min_psi)) {
    return;
  }

  std::cout << "    phase-bin RMSE [psi range: 3D / plane, samples]"
            << std::endl;

  for (std::size_t bin = 0; bin < bin_count; ++bin) {
    const double left = min_psi + (max_psi - min_psi) *
                                      static_cast<double>(bin) /
                                      static_cast<double>(bin_count);
    const double right = min_psi + (max_psi - min_psi) *
                                       static_cast<double>(bin + 1) /
                                       static_cast<double>(bin_count);

    std::cout << "      [" << std::fixed << std::setprecision(1) << left << ", "
              << right << "]: ";

    if (diagnostics.phase_bin_sample_count[bin] <= 0) {
      std::cout << "n/a" << std::endl;
      continue;
    }

    std::cout << std::setprecision(5) << diagnostics.phase_bin_rmse_3d_mm[bin]
              << " / " << diagnostics.phase_bin_plane_rmse_mm[bin] << " mm, "
              << diagnostics.phase_bin_sample_count[bin] << std::endl;
  }
}

double computeHarmonicPlaneOffsetRmse(
    const MsmCalibrationResult& result,
    const std::vector<DiscretePlane>& reference_planes,
    double* out_max_abs_mm = nullptr) {
  double weighted_sum_sq = 0.0;
  double weight_sum = 0.0;
  double max_abs_mm = 0.0;

  for (const auto& plane : reference_planes) {
    if (!plane.valid) {
      continue;
    }

    cv::Vec3d model_normal;
    double model_d = 0.0;
    result.evaluatePlane(plane.psi, model_normal, model_d);

    if (model_normal.dot(plane.normal) < 0.0) {
      model_normal = -model_normal;
      model_d = -model_d;
    }

    // reference_planes already use the continuous normal model, so d
    // difference is the orthogonal plane offset error.
    const double residual = model_d - plane.d;
    const double weight = std::max(plane.confidence, 1e-8);

    weighted_sum_sq += weight * residual * residual;
    weight_sum += weight;
    max_abs_mm = std::max(max_abs_mm, std::abs(residual));
  }

  if (out_max_abs_mm != nullptr) {
    *out_max_abs_mm = max_abs_mm;
  }

  return (weight_sum > 0.0) ? std::sqrt(weighted_sum_sq / weight_sum) : 0.0;
}

void printNormalAxisDiagnostics(const std::vector<DiscretePlane>& planes,
                                const cv::Vec3d& axis) {
  double weighted_sum_sq = 0.0;
  double weight_sum = 0.0;
  double max_abs = 0.0;

  for (const auto& plane : planes) {
    if (!plane.valid) {
      continue;
    }

    const double axial_component = plane.normal.dot(axis);
    const double weight = std::max(plane.confidence, 1e-8);
    weighted_sum_sq += weight * axial_component * axial_component;
    weight_sum += weight;
    max_abs = std::max(max_abs, std::abs(axial_component));
  }

  const double rms =
      (weight_sum > 0.0) ? std::sqrt(weighted_sum_sq / weight_sum) : 0.0;

  std::cout << "  observed normal axial-component RMS: " << rms
            << ", max: " << max_abs << std::endl;
}

}  // namespace

double RationalAngleModel::evaluateBase(double psi) const {
  if (!valid) {
    return 0.0;
  }

  const double delta_psi = psi - psi_ref;
  const double denominator = b0 + b1 * delta_psi;

  if (!std::isfinite(denominator) || std::abs(denominator) < 1e-12) {
    return 0.0;
  }

  return std::atan2(delta_psi, denominator);
}

double RationalAngleModel::evaluate(double psi) const {
  return evaluateBase(psi);
}

void HarmonicDriftModel::evaluate(double theta_rad, double& out_delta_u,
                                  double& out_delta_v) const {
  out_delta_u = 0.0;
  out_delta_v = 0.0;

  if (!valid || order <= 0) {
    return;
  }

  for (int k = 1; k <= order; ++k) {
    const int idx = 2 * (k - 1);
    const double ang = static_cast<double>(k) * theta_rad;
    const double cos_basis = std::cos(ang) - 1.0;
    const double sin_basis = std::sin(ang);

    if (idx + 1 < static_cast<int>(beta_u.size())) {
      out_delta_u += beta_u[idx] * cos_basis + beta_u[idx + 1] * sin_basis;
    }
    if (idx + 1 < static_cast<int>(beta_v.size())) {
      out_delta_v += beta_v[idx] * cos_basis + beta_v[idx + 1] * sin_basis;
    }
  }
}

void MsmCalibrationResult::evaluatePlane(double psi, cv::Vec3d& out_normal,
                                         double& out_d) const {
  const double theta = angle_model.evaluate(psi);

  cv::Mat R_theta;
  const cv::Vec3d rvec = nominal_axis_w * theta;
  cv::Rodrigues(rvec, R_theta);

  const cv::Mat n0_mat = (cv::Mat_<double>(3, 1) << ref_normal_n0[0],
                          ref_normal_n0[1], ref_normal_n0[2]);
  const cv::Mat n_mat = R_theta * n0_mat;

  out_normal = cv::normalize(
      cv::Vec3d(n_mat.at<double>(0), n_mat.at<double>(1), n_mat.at<double>(2)));

  double delta_u = 0.0;
  double delta_v = 0.0;
  if (harmonic_drift.valid && harmonic_drift.order > 0) {
    harmonic_drift.evaluate(theta, delta_u, delta_v);
  }

  const cv::Vec3d S = nominal_center_s0 + delta_u * basis_u + delta_v * basis_v;
  out_d = -out_normal.dot(S);
}

BoardPlane computeBoardPlane(const cv::Mat& rvec_or_R, const cv::Mat& t_mm) {
  if (rvec_or_R.empty() || t_mm.empty()) {
    throw std::invalid_argument("Camera extrinsics must not be empty.");
  }

  cv::Mat R_64;
  cv::Mat t_64;
  rvec_or_R.convertTo(R_64, CV_64F);
  t_mm.convertTo(t_64, CV_64F);

  cv::Mat R;
  if (R_64.rows == 3 && R_64.cols == 3) {
    R = R_64;
  } else {
    cv::Rodrigues(R_64, R);
  }

  cv::Vec3d n(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
  n = cv::normalize(n);
  if (n[2] < 0.0) {
    n = -n;
  }

  cv::Mat t_col = t_64.reshape(1, 3);
  const cv::Vec3d t(t_col.at<double>(0), t_col.at<double>(1),
                    t_col.at<double>(2));

  // Project convention: all geometric lengths are millimetres.
  // No automatic metre/mm guessing is allowed here.
  const double d = -n.dot(t);
  return {n, d};
}

std::pair<double, double> autoDetectValidPhaseRange(
    const std::vector<cv::Mat>& phase_maps, int min_covisible_poses) {
  if (phase_maps.empty()) return {0.0, 0.0};

  struct PoseInterval {
    double p_low = 0.0;
    double p_high = 0.0;
  };

  std::vector<PoseInterval> intervals;
  intervals.reserve(phase_maps.size());

  for (const auto& pmap : phase_maps) {
    if (pmap.empty()) continue;

    cv::Mat p64;
    if (pmap.channels() > 1) {
      cv::extractChannel(pmap, p64, 0);
    } else {
      p64 = pmap;
    }
    p64.convertTo(p64, CV_64F);

    std::vector<double> valid_vals;
    valid_vals.reserve(p64.rows * p64.cols / 16);

    for (int y = 0; y < p64.rows; y += 4) {
      const double* r_ptr = p64.ptr<double>(y);
      for (int x = 0; x < p64.cols; x += 4) {
        const double v = r_ptr[x];
        if (std::isfinite(v)) {
          valid_vals.push_back(v);
        }
      }
    }

    if (valid_vals.size() < 2000) continue;

    std::sort(valid_vals.begin(), valid_vals.end());
    const std::size_t idx_03 =
        static_cast<std::size_t>(valid_vals.size() * 0.03);
    const std::size_t idx_97 =
        static_cast<std::size_t>(valid_vals.size() * 0.97);

    intervals.push_back({valid_vals[idx_03], valid_vals[idx_97]});
  }

  if (intervals.empty()) return {0.0, 0.0};

  double global_min = std::numeric_limits<double>::infinity();
  double global_max = -std::numeric_limits<double>::infinity();
  for (const auto& interval : intervals) {
    global_min = std::min(global_min, interval.p_low);
    global_max = std::max(global_max, interval.p_high);
  }

  const int required_poses = std::max(
      2, std::min(min_covisible_poses, static_cast<int>(intervals.size())));

  constexpr double step = 0.5;
  double best_start = 0.0;
  double best_end = 0.0;
  double cur_start = -1.0;
  double max_len = 0.0;

  for (double psi = global_min; psi <= global_max; psi += step) {
    int covisible_count = 0;
    for (const auto& interval : intervals) {
      if (psi >= interval.p_low && psi <= interval.p_high) {
        ++covisible_count;
      }
    }

    if (covisible_count >= required_poses) {
      if (cur_start < 0.0) cur_start = psi;
    } else if (cur_start >= 0.0) {
      const double len = (psi - step) - cur_start;
      if (len > max_len) {
        max_len = len;
        best_start = cur_start;
        best_end = psi - step;
      }
      cur_start = -1.0;
    }
  }

  if (cur_start >= 0.0) {
    const double len = global_max - cur_start;
    if (len > max_len) {
      best_start = cur_start;
      best_end = global_max;
    }
  }

  best_start += 2.0;
  best_end -= 2.0;
  if (best_end <= best_start) {
    return {global_min, global_max};
  }

  return {best_start, best_end};
}

std::vector<cv::Point2d> extractIsoPhaseSubpixels(
    const cv::Mat& phase_map, double target_psi, const cv::Mat& mask,
    double grad_threshold, int fit_half_window, double max_grad_threshold,
    double max_lateral_jump_px) {
  if (phase_map.empty()) return {};

  if (fit_half_window < 1) {
    throw std::invalid_argument("fit_half_window must be at least 1.");
  }

  const double min_abs_gradient = std::max(grad_threshold, 1e-9);
  if (!(max_grad_threshold > min_abs_gradient)) {
    throw std::invalid_argument(
        "max_grad_threshold must be greater than grad_threshold.");
  }

  cv::Mat phase_f64;
  if (phase_map.channels() > 1) {
    cv::extractChannel(phase_map, phase_f64, 0);
  } else {
    phase_f64 = phase_map;
  }
  if (phase_f64.type() != CV_64F) {
    phase_f64.convertTo(phase_f64, CV_64F);
  }

  const int rows = phase_f64.rows;
  const int cols = phase_f64.cols;
  const bool has_mask = !mask.empty() && mask.size() == phase_f64.size();

  std::vector<cv::Point2d> raw_subpixels;
  raw_subpixels.reserve(rows);

  for (int y = 0; y < rows; ++y) {
    const double* row_ptr = phase_f64.ptr<double>(y);
    const uchar* mask_ptr = has_mask ? mask.ptr<uchar>(y) : nullptr;

    LocalPhaseFit best_fit;

    for (int x = 0; x < cols - 1; ++x) {
      if (has_mask && (!mask_ptr[x] || !mask_ptr[x + 1])) {
        continue;
      }

      const double p0 = row_ptr[x];
      const double p1 = row_ptr[x + 1];
      if (!std::isfinite(p0) || !std::isfinite(p1)) {
        continue;
      }

      const bool crossing = (p0 <= target_psi && target_psi <= p1) ||
                            (p1 <= target_psi && target_psi <= p0);
      if (!crossing || std::abs(p1 - p0) < min_abs_gradient) {
        continue;
      }

      const LocalPhaseFit candidate =
          fitLocalPhaseLine(phase_f64, y, x, target_psi, mask, fit_half_window,
                            min_abs_gradient, max_grad_threshold);

      if (!candidate.valid) {
        continue;
      }

      if (!best_fit.valid || candidate.rmse < best_fit.rmse) {
        best_fit = candidate;
      }
    }

    if (best_fit.valid) {
      raw_subpixels.emplace_back(best_fit.x, static_cast<double>(y));
    }
  }

  if (raw_subpixels.size() < 10 || max_lateral_jump_px <= 0.0) {
    return raw_subpixels;
  }

  // Suppress isolated row-wise x outliers while preserving the smooth
  // iso-phase curve.
  std::vector<cv::Point2d> clean_subpixels;
  clean_subpixels.reserve(raw_subpixels.size());

  constexpr int kRowMedianHalfWindow = 5;
  for (std::size_t i = 0; i < raw_subpixels.size(); ++i) {
    std::vector<double> local_xs;
    local_xs.reserve(2 * kRowMedianHalfWindow + 1);

    for (int w = -kRowMedianHalfWindow; w <= kRowMedianHalfWindow; ++w) {
      const int idx = static_cast<int>(i) + w;
      if (idx >= 0 && idx < static_cast<int>(raw_subpixels.size())) {
        local_xs.push_back(raw_subpixels[static_cast<std::size_t>(idx)].x);
      }
    }

    const std::size_t mid = local_xs.size() / 2;
    std::nth_element(local_xs.begin(),
                     local_xs.begin() + static_cast<std::ptrdiff_t>(mid),
                     local_xs.end());
    const double median_x = local_xs[mid];

    if (std::abs(raw_subpixels[i].x - median_x) <= max_lateral_jump_px) {
      clean_subpixels.push_back(raw_subpixels[i]);
    }
  }

  return clean_subpixels;
}

std::vector<cv::Vec3d> projectSubpixelsToBoard(
    const std::vector<cv::Point2d>& subpixels, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const BoardPlane& board_plane,
    double min_depth_mm, double max_depth_mm) {
  if (subpixels.empty()) return {};

  std::vector<cv::Point2d> norm_points;
  cv::undistortPoints(subpixels, norm_points, camera_matrix, dist_coeffs);

  std::vector<cv::Vec3d> points;
  points.reserve(subpixels.size());

  for (const auto& pt : norm_points) {
    cv::Vec3d ray(pt.x, pt.y, 1.0);
    ray = cv::normalize(ray);

    const double denom = board_plane.normal.dot(ray);
    if (std::abs(denom) < 1e-4) {
      continue;
    }

    const double depth = -board_plane.d / denom;
    if (!std::isfinite(depth) || depth < min_depth_mm || depth > max_depth_mm) {
      continue;
    }

    points.push_back(ray * depth);
  }

  return points;
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

double evaluateMsmReconstruction(const MsmCalibrationResult& result,
                                 const cv::Mat& phase_map,
                                 const cv::Mat& camera_matrix,
                                 const cv::Mat& dist_coeffs,
                                 const BoardPlane& board_plane, double min_psi,
                                 double max_psi, int pixel_stride) {
  return evaluateReconstructionDetailed(result, phase_map, camera_matrix,
                                        dist_coeffs, board_plane, min_psi,
                                        max_psi, pixel_stride)
      .rmse_3d_mm;
}

MsmCalibrationResult calibrateMsm(const MsmCalibrationConfig& config,
                                  const CameraCalibrationResult& camera_calib,
                                  const std::vector<cv::Mat>& all_phase_maps) {
  std::cout << "\nStarting MSM calibration pipeline..." << std::endl;

  const std::size_t train_count = config.train_poses.size();
  std::vector<BoardPlane> train_board_planes;
  std::vector<cv::Mat> train_phase_maps;

  train_board_planes.reserve(train_count);
  train_phase_maps.reserve(train_count);

  std::cout << "[Step 1] Preparing board planes for " << train_count
            << " training poses..." << std::endl;

  for (std::size_t i = 0; i < train_count; ++i) {
    const int pid = config.train_poses[i];

    if (pid < 0 ||
        pid >= static_cast<int>(camera_calib.rotation_vectors.size()) ||
        pid >= static_cast<int>(camera_calib.translation_vectors.size()) ||
        pid >= static_cast<int>(all_phase_maps.size()) ||
        all_phase_maps[static_cast<std::size_t>(pid)].empty()) {
      throw std::runtime_error("Invalid or missing training pose: " +
                               std::to_string(pid));
    }

    const auto board_plane =
        computeBoardPlane(camera_calib.rotation_vectors[pid],
                          camera_calib.translation_vectors[pid]);

    train_board_planes.push_back(board_plane);
    train_phase_maps.push_back(all_phase_maps[static_cast<std::size_t>(pid)]);

    std::cout << "  Pose " << std::setw(2) << pid << ": board normal=["
              << board_plane.normal[0] << ", " << board_plane.normal[1] << ", "
              << board_plane.normal[2] << "], distance d=" << board_plane.d
              << " mm" << std::endl;
  }

  double eff_min_psi = config.options.min_psi;
  double eff_max_psi = config.options.max_psi;

  if (config.options.auto_phase_range) {
    std::cout << "[Step 2] Auto-detecting multi-view co-visible phase range "
              << "(min_covisible_poses=" << config.options.min_covisible_poses
              << ") ..." << std::endl;

    const auto auto_range = autoDetectValidPhaseRange(
        train_phase_maps, config.options.min_covisible_poses);
    eff_min_psi = auto_range.first;
    eff_max_psi = auto_range.second;

    std::cout << "  Auto-detected co-visible range: [" << eff_min_psi << ", "
              << eff_max_psi << "] rad (span: " << eff_max_psi - eff_min_psi
              << " rad)" << std::endl;
  } else {
    std::cout << "[Step 2] Using configured phase range: [" << eff_min_psi
              << ", " << eff_max_psi << "] rad" << std::endl;
  }

  if (!(eff_max_psi > eff_min_psi)) {
    throw std::runtime_error(
        "Invalid effective phase range after range detection.");
  }

  auto collect_points_by_pose = [&](double target_psi) {
    PosePointGroups points_by_pose(train_count);

    for (std::size_t i = 0; i < train_count; ++i) {
      const auto subpixels =
          extractIsoPhaseSubpixels(train_phase_maps[i], target_psi, cv::Mat(),
                                   config.options.min_local_phase_gradient,
                                   config.options.iso_fit_half_window,
                                   config.options.max_local_phase_gradient,
                                   config.options.max_lateral_jump_px);

      if (subpixels.empty()) {
        continue;
      }

      auto points =
          projectSubpixelsToBoard(subpixels, camera_calib.camera_matrix,
                                  camera_calib.distortion_coefficients,
                                  train_board_planes[i], 120.0, 250.0);

      if (static_cast<int>(points.size()) <
          config.options.min_points_per_pose) {
        continue;
      }

      points_by_pose[i] = std::move(points);
    }

    return points_by_pose;
  };

  auto fit_plane_set = [&](const std::vector<double>& target_phases,
                           bool print_examples,
                           PlaneObservationSets* out_observations,
                           int required_pose_count) {
    std::vector<DiscretePlane> planes;
    planes.reserve(target_phases.size());

    if (out_observations != nullptr) {
      out_observations->clear();
      out_observations->reserve(target_phases.size());
    }

    auto fit_options = config.options;
    fit_options.min_covisible_poses = std::max(1, required_pose_count);

    for (std::size_t p_idx = 0; p_idx < target_phases.size(); ++p_idx) {
      const double target_psi = target_phases[p_idx];
      auto points_by_pose = collect_points_by_pose(target_psi);

      auto plane = fitPlaneRobustTLS(points_by_pose, target_psi, fit_options);
      planes.push_back(plane);

      if (out_observations != nullptr) {
        out_observations->push_back(std::move(points_by_pose));
      }

      if (print_examples && (p_idx == 0 || p_idx == target_phases.size() / 2 ||
                             p_idx + 1 == target_phases.size())) {
        std::cout << "  Plane " << std::setw(2) << p_idx
                  << " (psi=" << std::fixed << std::setprecision(2)
                  << target_psi << "): poses=" << plane.pose_count
                  << ", pts=" << plane.point_count
                  << ", rms=" << std::setprecision(5) << plane.rms_mm << " mm"
                  << ", spread=" << plane.spread_ratio
                  << ", thickness=" << plane.thickness_ratio
                  << ", valid=" << (plane.valid ? "YES" : "NO") << std::endl;
      }
    }

    enforceNormalConsistency(planes);
    normalizePlaneConfidences(planes);
    return planes;
  };

  const double ref_psi = 0.5 * (eff_min_psi + eff_max_psi);
  const int final_plane_count = std::max(10, config.options.plane_count);
  const int coarse_plane_count = std::clamp(final_plane_count / 2, 24, 40);

  std::cout << "[Step 3] Estimating phase-to-angle mapping..." << std::endl;

  std::vector<DiscretePlane> coarse_planes;
  double coarse_min_psi = eff_min_psi;
  double coarse_max_psi = eff_max_psi;
  int coarse_required_poses = config.options.min_covisible_poses;
  bool initialization_found = false;

  const int maximum_initial_pose_count = static_cast<int>(train_count);
  const int minimum_initial_pose_count = std::max(
      2,
      std::min(config.options.min_covisible_poses, maximum_initial_pose_count));

  for (int required_poses = maximum_initial_pose_count;
       required_poses >= minimum_initial_pose_count; --required_poses) {
    const auto candidate_range =
        autoDetectValidPhaseRange(train_phase_maps, required_poses);

    if (!(candidate_range.second > candidate_range.first)) {
      continue;
    }

    const auto candidate_phases = makeUniformSamples(
        candidate_range.first, candidate_range.second, coarse_plane_count);

    auto candidate_planes =
        fit_plane_set(candidate_phases, false, nullptr, required_poses);

    int valid_count = 0;
    for (const auto& plane : candidate_planes) {
      if (plane.valid) {
        ++valid_count;
      }
    }

    std::cout << "  initialization candidate: " << required_poses << "/"
              << train_count << " co-visible poses, range=[" << std::fixed
              << std::setprecision(3) << candidate_range.first << ", "
              << candidate_range.second << "] rad"
              << ", valid planes=" << valid_count << "/" << coarse_plane_count
              << std::endl;

    if (valid_count >= 6) {
      coarse_planes = std::move(candidate_planes);
      coarse_min_psi = candidate_range.first;
      coarse_max_psi = candidate_range.second;
      coarse_required_poses = required_poses;
      initialization_found = true;
      break;
    }
  }

  if (!initialization_found) {
    std::cout << "  No stable initialization range was found. "
              << "Plane diagnostics for the configured overlap are:"
              << std::endl;

    const auto fallback_phases =
        makeUniformSamples(eff_min_psi, eff_max_psi, coarse_plane_count);
    auto fallback_planes = fit_plane_set(fallback_phases, false, nullptr,
                                         config.options.min_covisible_poses);
    printPlaneQualityDiagnostics(fallback_planes, config.options);

    throw std::runtime_error(
        "Insufficient valid initial planes for angle estimation.");
  }

  std::cout << "  selected initialization range: [" << coarse_min_psi << ", "
            << coarse_max_psi << "] rad using " << coarse_required_poses << "/"
            << train_count << " co-visible poses" << std::endl;

  const double coarse_ref_psi = 0.5 * (coarse_min_psi + coarse_max_psi);

  cv::Vec3d coarse_w;
  cv::Vec3d coarse_s0;
  cv::Vec3d coarse_n0;
  cv::Vec3d coarse_u;
  cv::Vec3d coarse_v;

  if (!solveNominalRotationGeometry(coarse_planes, coarse_ref_psi, coarse_w,
                                    coarse_s0, coarse_n0, coarse_u, coarse_v)) {
    throw std::runtime_error("Failed to solve initial rotation geometry.");
  }

  const auto coarse_thetas =
      computeRelativeAngles(coarse_planes, coarse_w, coarse_n0, true);
  const auto coarse_angle_model =
      fitRationalAngleModel(coarse_planes, coarse_thetas, coarse_ref_psi);

  if (!coarse_angle_model.valid) {
    throw std::runtime_error(
        "Failed to estimate initial phase-to-angle model.");
  }

  RationalAngleModel sampling_model = coarse_angle_model;
  MsmCalibrationResult result;
  std::vector<DiscretePlane> discrete_planes;
  PlaneObservationSets final_observations;
  std::vector<double> thetas;
  std::vector<double> previous_phases;

  const int max_resampling_iterations =
      std::max(1, config.options.angle_resampling_iterations);

  std::cout << "[Step 4] Fitting " << final_plane_count
            << " planes uniformly in optical angle..." << std::endl;

  for (int iteration = 0; iteration < max_resampling_iterations; ++iteration) {
    const double current_theta_min = sampling_model.evaluate(eff_min_psi);
    const double current_theta_max = sampling_model.evaluate(eff_max_psi);

    if (!std::isfinite(current_theta_min) ||
        !std::isfinite(current_theta_max) ||
        std::abs(current_theta_max - current_theta_min) < 1e-8) {
      throw std::runtime_error(
          "Invalid optical-angle range during resampling.");
    }

    const auto target_thetas = makeUniformSamples(
        current_theta_min, current_theta_max, final_plane_count);

    std::vector<double> refined_phases;
    refined_phases.reserve(target_thetas.size());

    for (const double theta : target_thetas) {
      const double psi =
          invertAngleModel(sampling_model, theta, eff_min_psi, eff_max_psi);
      if (!std::isfinite(psi)) {
        throw std::runtime_error(
            "Failed to invert phase-to-angle model during "
            "uniform-angle resampling.");
      }
      refined_phases.push_back(psi);
    }

    double max_phase_update = 0.0;
    if (!previous_phases.empty() &&
        previous_phases.size() == refined_phases.size()) {
      for (std::size_t i = 0; i < refined_phases.size(); ++i) {
        max_phase_update = std::max(
            max_phase_update, std::abs(refined_phases[i] - previous_phases[i]));
      }
    }

    PlaneObservationSets observations;
    auto planes = fit_plane_set(
        refined_phases, iteration + 1 == max_resampling_iterations,
        &observations, config.options.min_covisible_poses);

    int valid_count = 0;
    for (const auto& plane : planes) {
      if (plane.valid) {
        ++valid_count;
      }
    }

    if (valid_count < 6) {
      throw std::runtime_error(
          "Insufficient valid planes for MSM calibration.");
    }

    MsmCalibrationResult iteration_result;
    if (!solveNominalRotationGeometry(
            planes, ref_psi, iteration_result.nominal_axis_w,
            iteration_result.nominal_center_s0, iteration_result.ref_normal_n0,
            iteration_result.basis_u, iteration_result.basis_v)) {
      throw std::runtime_error("Failed to solve nominal rotation geometry.");
    }

    auto iteration_thetas =
        computeRelativeAngles(planes, iteration_result.nominal_axis_w,
                              iteration_result.ref_normal_n0, true);

    iteration_result.angle_model =
        fitRationalAngleModel(planes, iteration_thetas, ref_psi);

    if (!iteration_result.angle_model.valid) {
      throw std::runtime_error("Failed to fit phase-to-angle model.");
    }

    double iteration_max_angle_error = 0.0;
    const double iteration_angle_rmse = computeAngleModelRmse(
        planes, iteration_thetas, iteration_result.angle_model,
        &iteration_max_angle_error);

    std::cout << "  resampling iteration " << iteration + 1
              << ": valid planes=" << valid_count << "/" << final_plane_count
              << ", angle RMSE=" << iteration_angle_rmse * 1000.0 << " mrad";

    if (!previous_phases.empty()) {
      std::cout << ", max phase update=" << max_phase_update << " rad";
    }
    std::cout << std::endl;

    discrete_planes = std::move(planes);
    final_observations = std::move(observations);
    thetas = std::move(iteration_thetas);
    result = iteration_result;

    const bool converged =
        !previous_phases.empty() &&
        max_phase_update <= config.options.angle_resampling_tolerance;

    previous_phases = std::move(refined_phases);
    sampling_model = result.angle_model;

    if (converged) {
      std::cout << "  optical-angle sampling converged." << std::endl;
      break;
    }
  }

  printPlaneQualityDiagnostics(discrete_planes, config.options);

  double max_angle_error = 0.0;
  const double angle_rmse = computeAngleModelRmse(
      discrete_planes, thetas, result.angle_model, &max_angle_error);

  const double theta_min = result.angle_model.evaluate(eff_min_psi);
  const double theta_max = result.angle_model.evaluate(eff_max_psi);

  double min_denominator = std::numeric_limits<double>::infinity();
  double max_denominator = -std::numeric_limits<double>::infinity();

  for (const auto& plane : discrete_planes) {
    if (!plane.valid) {
      continue;
    }
    const double delta_psi = plane.psi - result.angle_model.psi_ref;
    const double denominator =
        result.angle_model.b0 + result.angle_model.b1 * delta_psi;
    min_denominator = std::min(min_denominator, denominator);
    max_denominator = std::max(max_denominator, denominator);
  }

  std::cout << "\nAngle model diagnostics" << std::endl;
  std::cout << "  optical angle range: [" << theta_min << ", " << theta_max
            << "] rad" << std::endl;
  std::cout << "  weighted RMSE: " << angle_rmse * 1000.0 << " mrad"
            << std::endl;
  std::cout << "  maximum absolute angle residual: " << max_angle_error * 1000.0
            << " mrad" << std::endl;
  std::cout << "  denominator range: [" << min_denominator << ", "
            << max_denominator << "]" << std::endl;
  printNormalAxisDiagnostics(discrete_planes, result.nominal_axis_w);

  printPosePlaneResidualDiagnostics(
      "Observed discrete-plane residuals by training pose", discrete_planes,
      final_observations, config.train_poses);

  std::vector<DiscretePlane> consistent_planes;
  consistent_planes.reserve(discrete_planes.size());

  std::vector<double> model_thetas(discrete_planes.size(), 0.0);

  for (std::size_t i = 0; i < discrete_planes.size(); ++i) {
    const auto& source_plane = discrete_planes[i];

    if (!source_plane.valid) {
      consistent_planes.push_back(source_plane);
      continue;
    }

    const double theta = result.angle_model.evaluate(source_plane.psi);
    model_thetas[i] = theta;

    cv::Vec3d model_normal =
        rotateAroundAxis(result.ref_normal_n0, result.nominal_axis_w, theta);
    model_normal =
        cv::normalize(model_normal - model_normal.dot(result.nominal_axis_w) *
                                         result.nominal_axis_w);

    auto plane =
        fitOffsetForFixedNormal(final_observations[i], source_plane.psi,
                                model_normal, source_plane, config.options);

    consistent_planes.push_back(plane);
  }

  normalizePlaneConfidences(consistent_planes);

  printPlaneQualityDiagnostics(consistent_planes, config.options);
  printPosePlaneResidualDiagnostics(
      "Continuous-normal plane residuals by training pose", consistent_planes,
      final_observations, config.train_poses);

  cv::Vec3d consistent_center;
  if (!solveCenterFromPlanes(consistent_planes, result.basis_u, result.basis_v,
                             consistent_center)) {
    throw std::runtime_error("Failed to solve model-consistent scan center.");
  }
  result.nominal_center_s0 = consistent_center;

  cv::Vec3d reference_center;
  result.harmonic_drift = fitHarmonicDriftModel(
      consistent_planes, model_thetas, result.nominal_axis_w,
      result.nominal_center_s0, result.basis_u, result.basis_v,
      reference_center, config.options.harmonic_order,
      config.options.harmonic_regularization);
  result.nominal_center_s0 = reference_center;

  double harmonic_max_offset_mm = 0.0;
  const double harmonic_offset_rmse_mm = computeHarmonicPlaneOffsetRmse(
      result, consistent_planes, &harmonic_max_offset_mm);

  std::cout << "\nContinuous plane model diagnostics" << std::endl;
  std::cout << "  harmonic plane-offset RMSE: " << std::fixed
            << std::setprecision(5) << harmonic_offset_rmse_mm << " mm"
            << std::endl;
  std::cout << "  maximum absolute plane-offset residual: "
            << harmonic_max_offset_mm << " mm" << std::endl;

  double sum_train_pose_sq = 0.0;
  int train_eval_count = 0;

  std::cout << "\n[Step 5] Reconstruction diagnostics on training poses:"
            << std::endl;

  for (std::size_t i = 0; i < train_count; ++i) {
    const int pid = config.train_poses[i];

    const auto diagnostics = evaluateReconstructionDetailed(
        result, train_phase_maps[i], camera_calib.camera_matrix,
        camera_calib.distortion_coefficients, train_board_planes[i],
        eff_min_psi, eff_max_psi, 8);

    const double pose_error = diagnostics.rmse_3d_mm;

    if (pose_error > 0.0 && std::isfinite(pose_error)) {
      sum_train_pose_sq += pose_error * pose_error;
      ++train_eval_count;

      std::cout << "  train pose " << std::setw(2) << pid
                << ": 3D RMSE=" << std::fixed << std::setprecision(5)
                << pose_error << " mm"
                << ", model-plane RMSE=" << diagnostics.model_plane_rmse_mm
                << " mm"
                << ", |n.r| median=" << diagnostics.ray_plane_denom_median
                << ", p05=" << diagnostics.ray_plane_denom_p05
                << ", amplification=" << diagnostics.amplification << std::endl;
    }
  }

  result.train_rmse_mm =
      (train_eval_count > 0)
          ? std::sqrt(sum_train_pose_sq / static_cast<double>(train_eval_count))
          : 0.0;

  double sum_test_pose_sq = 0.0;
  int test_eval_count = 0;

  if (!config.test_poses.empty()) {
    std::cout << "\n[Step 6] Reconstruction diagnostics on test poses:"
              << std::endl;
  }

  for (const int test_pid : config.test_poses) {
    if (test_pid < 0 || test_pid >= static_cast<int>(all_phase_maps.size()) ||
        test_pid >= static_cast<int>(camera_calib.rotation_vectors.size()) ||
        test_pid >= static_cast<int>(camera_calib.translation_vectors.size()) ||
        all_phase_maps[static_cast<std::size_t>(test_pid)].empty()) {
      std::cout << "  test pose " << test_pid
                << ": skipped (invalid or missing data)" << std::endl;
      continue;
    }

    const auto test_board_plane =
        computeBoardPlane(camera_calib.rotation_vectors[test_pid],
                          camera_calib.translation_vectors[test_pid]);

    const auto diagnostics = evaluateReconstructionDetailed(
        result, all_phase_maps[static_cast<std::size_t>(test_pid)],
        camera_calib.camera_matrix, camera_calib.distortion_coefficients,
        test_board_plane, eff_min_psi, eff_max_psi, 4,
        config.options.diagnostic_phase_bins);

    const double pose_error = diagnostics.rmse_3d_mm;

    if (pose_error > 0.0 && std::isfinite(pose_error)) {
      sum_test_pose_sq += pose_error * pose_error;
      ++test_eval_count;

      std::cout << "  test pose " << std::setw(2) << test_pid
                << ": 3D RMSE=" << std::fixed << std::setprecision(5)
                << pose_error << " mm"
                << ", model-plane RMSE=" << diagnostics.model_plane_rmse_mm
                << " mm"
                << ", |n.r| median=" << diagnostics.ray_plane_denom_median
                << ", p05=" << diagnostics.ray_plane_denom_p05
                << ", amplification=" << diagnostics.amplification << std::endl;
      printPhaseBinDiagnostics(diagnostics, eff_min_psi, eff_max_psi);
    }
  }

  result.test_rmse_mm =
      (test_eval_count > 0)
          ? std::sqrt(sum_test_pose_sq / static_cast<double>(test_eval_count))
          : 0.0;

  std::cout << "\n================ Calibration Summary ================"
            << std::endl;
  std::cout << "Nominal Axis (w): [" << result.nominal_axis_w[0] << ", "
            << result.nominal_axis_w[1] << ", " << result.nominal_axis_w[2]
            << "]" << std::endl;
  std::cout << "Nominal Center (S0): [" << result.nominal_center_s0[0] << ", "
            << result.nominal_center_s0[1] << ", "
            << result.nominal_center_s0[2] << "] mm" << std::endl;
  std::cout << "Angle Model: rational trend" << std::endl;
  std::cout << "  tan(theta_base) = dpsi / (" << result.angle_model.b0 << " + "
            << result.angle_model.b1 << "*dpsi), dpsi = psi - "
            << result.angle_model.psi_ref << std::endl;
  std::cout << "Harmonic Drift Order: " << result.harmonic_drift.order
            << " (valid=" << (result.harmonic_drift.valid ? "YES" : "NO") << ")"
            << std::endl;

  if (result.harmonic_drift.valid && result.harmonic_drift.order > 0) {
    std::cout << "Harmonic basis is anchored at the reference angle "
              << "(delta S(0) = 0)." << std::endl;
  }

  std::cout << "Closed-Loop TRAIN 3D RMSE: " << result.train_rmse_mm << " mm"
            << std::endl;
  std::cout << "Closed-Loop TEST 3D RMSE:  " << result.test_rmse_mm << " mm"
            << std::endl;
  std::cout << "=====================================================\n"
            << std::endl;

  return result;
}
bool saveMsmCalibrationResult(const std::string& file_path,
                              const MsmCalibrationResult& result) {
  cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "nominal_axis_w" << cv::Mat(result.nominal_axis_w);
  fs << "nominal_center_s0" << cv::Mat(result.nominal_center_s0);
  fs << "ref_normal_n0" << cv::Mat(result.ref_normal_n0);
  fs << "basis_u" << cv::Mat(result.basis_u);
  fs << "basis_v" << cv::Mat(result.basis_v);

  fs << "psi_ref" << result.angle_model.psi_ref;
  fs << "angle_model_b0" << result.angle_model.b0;
  fs << "angle_model_b1" << result.angle_model.b1;

  fs << "harmonic_order" << result.harmonic_drift.order;
  fs << "harmonic_basis" << "anchored_cos_minus_one";
  fs << "beta_u" << result.harmonic_drift.beta_u;
  fs << "beta_v" << result.harmonic_drift.beta_v;

  fs << "train_rmse_mm" << result.train_rmse_mm;
  fs << "test_rmse_mm" << result.test_rmse_mm;

  fs.release();
  return true;
}

MsmCalibrationResult loadMsmCalibrationResult(const std::string& file_path) {
  cv::FileStorage fs(file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Cannot open MSM result file: " + file_path);
  }

  MsmCalibrationResult result;
  cv::Mat w_mat;
  cv::Mat s0_mat;
  cv::Mat n0_mat;
  cv::Mat u_mat;
  cv::Mat v_mat;

  fs["nominal_axis_w"] >> w_mat;
  fs["nominal_center_s0"] >> s0_mat;
  fs["ref_normal_n0"] >> n0_mat;
  fs["basis_u"] >> u_mat;
  fs["basis_v"] >> v_mat;

  result.nominal_axis_w = cv::Vec3d(w_mat);
  result.nominal_center_s0 = cv::Vec3d(s0_mat);
  result.ref_normal_n0 = cv::Vec3d(n0_mat);
  result.basis_u = cv::Vec3d(u_mat);
  result.basis_v = cv::Vec3d(v_mat);

  fs["psi_ref"] >> result.angle_model.psi_ref;

  const cv::FileNode b0_node = fs["angle_model_b0"];
  const cv::FileNode b1_node = fs["angle_model_b1"];

  if (!b0_node.empty() && !b1_node.empty()) {
    b0_node >> result.angle_model.b0;
    b1_node >> result.angle_model.b1;
  } else {
    double old_a1 = 0.0;
    double old_a2 = 0.0;
    fs["angle_model_a1"] >> old_a1;
    fs["angle_model_a2"] >> old_a2;

    result.angle_model.b1 = old_a1;
    result.angle_model.b0 = old_a1 * result.angle_model.psi_ref + old_a2;
  }

  result.angle_model.valid = true;

  fs["harmonic_order"] >> result.harmonic_drift.order;

  std::vector<double> stored_beta_u;
  std::vector<double> stored_beta_v;
  fs["beta_u"] >> stored_beta_u;
  fs["beta_v"] >> stored_beta_v;

  std::string harmonic_basis;
  const cv::FileNode basis_node = fs["harmonic_basis"];
  if (!basis_node.empty()) {
    basis_node >> harmonic_basis;
  }

  const int order = result.harmonic_drift.order;
  const std::size_t anchored_size =
      static_cast<std::size_t>(std::max(0, 2 * order));
  const std::size_t legacy_size =
      static_cast<std::size_t>(std::max(0, 2 * order + 1));

  if (harmonic_basis == "anchored_cos_minus_one" &&
      stored_beta_u.size() == anchored_size &&
      stored_beta_v.size() == anchored_size) {
    result.harmonic_drift.beta_u = std::move(stored_beta_u);
    result.harmonic_drift.beta_v = std::move(stored_beta_v);
    result.harmonic_drift.valid = (order > 0);
  } else if (order > 0 && stored_beta_u.size() == legacy_size &&
             stored_beta_v.size() == legacy_size) {
    // Legacy representation:
    // delta = dc + sum(A_k cos(k*theta) + B_k sin(k*theta)).
    // Move delta(0) into the reference center and convert the remaining
    // coefficients to A_k*(cos(k*theta)-1) + B_k*sin(k*theta).
    double delta_u_at_zero = stored_beta_u[0];
    double delta_v_at_zero = stored_beta_v[0];

    result.harmonic_drift.beta_u.assign(anchored_size, 0.0);
    result.harmonic_drift.beta_v.assign(anchored_size, 0.0);

    for (int k = 1; k <= order; ++k) {
      const int legacy_idx = 1 + 2 * (k - 1);
      const int anchored_idx = 2 * (k - 1);

      const double au = stored_beta_u[legacy_idx];
      const double bu = stored_beta_u[legacy_idx + 1];
      const double av = stored_beta_v[legacy_idx];
      const double bv = stored_beta_v[legacy_idx + 1];

      delta_u_at_zero += au;
      delta_v_at_zero += av;

      result.harmonic_drift.beta_u[anchored_idx] = au;
      result.harmonic_drift.beta_u[anchored_idx + 1] = bu;
      result.harmonic_drift.beta_v[anchored_idx] = av;
      result.harmonic_drift.beta_v[anchored_idx + 1] = bv;
    }

    result.nominal_center_s0 +=
        delta_u_at_zero * result.basis_u + delta_v_at_zero * result.basis_v;
    result.harmonic_drift.valid = true;
  } else {
    result.harmonic_drift.beta_u.clear();
    result.harmonic_drift.beta_v.clear();
    result.harmonic_drift.valid = false;
  }

  fs["train_rmse_mm"] >> result.train_rmse_mm;
  fs["test_rmse_mm"] >> result.test_rmse_mm;

  fs.release();
  return result;
}

}  // namespace msm3d

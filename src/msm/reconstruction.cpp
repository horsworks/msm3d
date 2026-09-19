#include "msm3d/msm/reconstruction.hpp"

#include "internal.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace msm3d {

ReconstructionDiagnostics evaluateReconstructionDetailed(
    const MsmCalibrationResult& result, const cv::Mat& phase_map,
    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
    const BoardPlane& board_plane, double min_psi, double max_psi,
    int pixel_stride, int phase_bin_count) {
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

}  // namespace msm3d

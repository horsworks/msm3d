#include "msm3d/camera_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

namespace msm3d {
namespace {

void validateBoard(const CalibrationBoard& board) {
  if (board.columns <= 1 || board.rows <= 1) {
    throw std::invalid_argument(
        "Calibration board rows and columns must be greater than 1.");
  }

  if (!std::isfinite(board.spacing) || board.spacing <= 0.0) {
    throw std::invalid_argument(
        "Calibration board spacing must be positive and finite.");
  }
}

cv::Mat convertToGray(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("Calibration image must not be empty.");
  }

  if (image.channels() == 1) {
    return image;
  }

  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
  } else {
    throw std::invalid_argument("Unsupported calibration image format.");
  }

  return gray;
}

cv::Ptr<cv::SimpleBlobDetector> createCircleDetector(
    const CircleDetectorParameters& detector_params) {
  cv::SimpleBlobDetector::Params params;

  params.minThreshold = detector_params.min_threshold;
  params.maxThreshold = detector_params.max_threshold;
  params.thresholdStep = detector_params.threshold_step;
  params.minRepeatability = detector_params.min_repeatability;
  params.minDistBetweenBlobs = detector_params.min_dist_between_blobs;
  params.filterByArea = detector_params.filter_by_area;
  params.minArea = detector_params.min_area;
  params.maxArea = detector_params.max_area;
  params.filterByCircularity = detector_params.filter_by_circularity;
  params.minCircularity = detector_params.min_circularity;
  params.filterByConvexity = detector_params.filter_by_convexity;
  params.minConvexity = detector_params.min_convexity;
  params.filterByInertia = detector_params.filter_by_inertia;
  params.minInertiaRatio = detector_params.min_inertia_ratio;
  params.filterByColor = detector_params.filter_by_color;
  params.blobColor = detector_params.blob_color;

  return cv::SimpleBlobDetector::create(params);
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }

  q = std::clamp(q, 0.0, 1.0);
  const double position = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));

  std::nth_element(values.begin(), values.begin() + lower, values.end());
  const double lower_value = values[lower];

  if (upper == lower) {
    return lower_value;
  }

  std::nth_element(values.begin(), values.begin() + upper, values.end());
  const double upper_value = values[upper];
  const double t = position - static_cast<double>(lower);
  return lower_value + t * (upper_value - lower_value);
}

int makeCalibrationFlags(const CameraCalibrationOptions& options) {
  int flags = 0;

  if (options.fix_k3) {
    flags |= cv::CALIB_FIX_K3;
  }
  if (options.zero_tangent_distortion) {
    flags |= cv::CALIB_ZERO_TANGENT_DIST;
  }
  if (options.use_rational_model) {
    flags |= cv::CALIB_RATIONAL_MODEL;
  }

  return flags;
}

}  // namespace

std::vector<cv::Point3f> generateCalibrationObjectPoints(
    const CalibrationBoard& board) {
  validateBoard(board);

  std::vector<cv::Point3f> points;
  points.reserve(static_cast<std::size_t>(board.columns * board.rows));

  const bool is_asymmetric =
      (board.pattern == CalibrationPattern::kAsymmetricCircles);

  for (int row = 0; row < board.rows; ++row) {
    const double y = static_cast<double>(row) * board.spacing;
    for (int col = 0; col < board.columns; ++col) {
      const double x = is_asymmetric ? (2.0 * col + (row % 2)) * board.spacing
                                     : col * board.spacing;
      points.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.0F);
    }
  }

  return points;
}

CalibrationDetectionResult detectCalibrationPoints(
    const cv::Mat& image, const CalibrationBoard& board,
    const CircleDetectorParameters& detector_params) {
  validateBoard(board);

  const cv::Mat gray = convertToGray(image);
  CalibrationDetectionResult result;
  const cv::Size pattern_size(board.columns, board.rows);

  if (board.pattern == CalibrationPattern::kChessboard) {
    result.found = cv::findChessboardCorners(
        gray, pattern_size, result.points,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (result.found) {
      cv::cornerSubPix(
          gray, result.points, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                           30, 0.001));
    }
    return result;
  }

  const auto detector = createCircleDetector(detector_params);
  detector->detect(gray, result.blob_keypoints);

  int flags = cv::CALIB_CB_CLUSTERING;
  flags |= (board.pattern == CalibrationPattern::kSymmetricCircles)
               ? cv::CALIB_CB_SYMMETRIC_GRID
               : cv::CALIB_CB_ASYMMETRIC_GRID;

  result.found =
      cv::findCirclesGrid(gray, pattern_size, result.points, flags, detector);

  if (result.found) {
    return result;
  }

  cv::Mat inverted_gray;
  cv::bitwise_not(gray, inverted_gray);

  std::vector<cv::KeyPoint> inverted_keypoints;
  detector->detect(inverted_gray, inverted_keypoints);

  std::vector<cv::Point2f> inverted_points;
  const bool inverted_found = cv::findCirclesGrid(
      inverted_gray, pattern_size, inverted_points, flags, detector);

  if (inverted_found) {
    result.found = true;
    result.points = std::move(inverted_points);
    result.blob_keypoints = std::move(inverted_keypoints);
    return result;
  }

  const std::size_t expected_count =
      static_cast<std::size_t>(board.columns * board.rows);
  const auto orig_diff =
      std::abs(static_cast<long long>(result.blob_keypoints.size()) -
               static_cast<long long>(expected_count));
  const auto inv_diff =
      std::abs(static_cast<long long>(inverted_keypoints.size()) -
               static_cast<long long>(expected_count));

  if (inv_diff < orig_diff) {
    result.blob_keypoints = std::move(inverted_keypoints);
  }

  return result;
}

CameraCalibrationResult calibrateCamera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size, const CameraCalibrationOptions& options) {
  if (object_points.size() != image_points.size()) {
    throw std::invalid_argument(
        "Object points and image points must have the same number of views.");
  }

  if (object_points.size() < 3) {
    throw std::invalid_argument(
        "At least 3 valid calibration views are required.");
  }

  if (image_size.width <= 0 || image_size.height <= 0) {
    throw std::invalid_argument("Invalid calibration image size.");
  }

  for (std::size_t i = 0; i < object_points.size(); ++i) {
    if (object_points[i].empty() || image_points[i].empty()) {
      throw std::invalid_argument("Calibration point sets must not be empty.");
    }

    if (object_points[i].size() != image_points[i].size()) {
      throw std::invalid_argument(
          "Object points and image points must correspond one-to-one.");
    }
  }

  CameraCalibrationResult result;
  result.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  result.distortion_coefficients =
      cv::Mat::zeros(1, options.use_rational_model ? 8 : 5, CV_64F);

  const int calibration_flags = makeCalibrationFlags(options);
  const cv::TermCriteria criteria(
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 200, 1e-12);

  cv::Mat opencv_per_view_errors;
  result.rms = cv::calibrateCamera(
      object_points, image_points, image_size, result.camera_matrix,
      result.distortion_coefficients, result.rotation_vectors,
      result.translation_vectors, result.intrinsic_std_deviations,
      result.extrinsic_std_deviations, opencv_per_view_errors,
      calibration_flags, criteria);

  double total_error_sum = 0.0;
  double total_squared_error = 0.0;
  std::size_t total_point_count = 0;
  std::vector<double> all_point_errors;

  result.per_view_errors.reserve(object_points.size());
  result.per_view_mean_errors.reserve(object_points.size());
  result.per_view_p95_errors.reserve(object_points.size());
  result.per_view_max_errors.reserve(object_points.size());

  for (std::size_t i = 0; i < object_points.size(); ++i) {
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points[i], result.rotation_vectors[i],
                      result.translation_vectors[i], result.camera_matrix,
                      result.distortion_coefficients, projected_points);

    std::vector<double> view_errors;
    view_errors.reserve(projected_points.size());

    double view_error_sum = 0.0;
    double view_squared_error = 0.0;
    double view_max_error = 0.0;

    for (std::size_t j = 0; j < projected_points.size(); ++j) {
      const double dx =
          static_cast<double>(image_points[i][j].x - projected_points[j].x);
      const double dy =
          static_cast<double>(image_points[i][j].y - projected_points[j].y);
      const double error = std::sqrt(dx * dx + dy * dy);

      view_errors.push_back(error);
      all_point_errors.push_back(error);
      view_error_sum += error;
      view_squared_error += error * error;
      view_max_error = std::max(view_max_error, error);
    }

    const double point_count = static_cast<double>(view_errors.size());
    result.per_view_mean_errors.push_back(view_error_sum / point_count);
    result.per_view_errors.push_back(
        std::sqrt(view_squared_error / point_count));
    result.per_view_p95_errors.push_back(percentile(view_errors, 0.95));
    result.per_view_max_errors.push_back(view_max_error);

    total_error_sum += view_error_sum;
    total_squared_error += view_squared_error;
    total_point_count += view_errors.size();
  }

  if (total_point_count == 0) {
    throw std::runtime_error(
        "Camera calibration produced no reprojection observations.");
  }

  const double total_count = static_cast<double>(total_point_count);
  result.mean_reprojection_error = total_error_sum / total_count;
  result.reprojection_error = std::sqrt(total_squared_error / total_count);
  result.p95_reprojection_error = percentile(all_point_errors, 0.95);
  result.max_reprojection_error =
      all_point_errors.empty()
          ? 0.0
          : *std::max_element(all_point_errors.begin(), all_point_errors.end());

  return result;
}

}  // namespace msm3d

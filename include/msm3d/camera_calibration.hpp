#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace msm3d {

enum class CalibrationPattern {
  kChessboard,
  kSymmetricCircles,
  kAsymmetricCircles
};

struct CalibrationBoard {
  int columns = 0;
  int rows = 0;
  float spacing = 0.0f;
  CalibrationPattern pattern = CalibrationPattern::kChessboard;
};

struct CircleDetectorParameters {
  float min_threshold = 10.0f;
  float max_threshold = 220.0f;
  float threshold_step = 5.0f;
  std::size_t min_repeatability = 2;
  float min_dist_between_blobs = 10.0f;
  bool filter_by_area = true;
  float min_area = 25.0f;
  float max_area = 5000.0f;
  bool filter_by_circularity = false;
  float min_circularity = 0.8f;
  bool filter_by_convexity = false;
  float min_convexity = 0.95f;
  bool filter_by_inertia = true;
  float min_inertia_ratio = 0.1f;
  bool filter_by_color = true;
  unsigned char blob_color = 0;
};

struct CameraCalibrationOptions {
  bool fix_k3 = true;
  bool zero_tangent_distortion = false;
  bool use_rational_model = false;
};

struct CameraCalibrationConfig {
  std::string image_dir;
  CalibrationBoard board;
  CircleDetectorParameters circle_detector;
  CameraCalibrationOptions calibration;
  std::string result_file;
  std::string detection_dir;
  std::string blob_dir;
  bool save_detection_debug = false;
  bool save_blob_debug = false;
};

struct CalibrationDetectionResult {
  bool found = false;
  std::vector<cv::Point2f> points;
  std::vector<cv::KeyPoint> blob_keypoints;
};

struct CameraCalibrationResult {
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  std::vector<cv::Mat> rotation_vectors;
  std::vector<cv::Mat> translation_vectors;

  // OpenCV calibrateCamera() overall RMS return value.
  double rms = 0.0;

  // Backward-compatible field: Euclidean point-error RMS over all points.
  double reprojection_error = 0.0;

  // MATLAB cameraParameters.MeanReprojectionError-compatible statistic:
  // arithmetic mean of Euclidean reprojection distances.
  double mean_reprojection_error = 0.0;
  double p95_reprojection_error = 0.0;
  double max_reprojection_error = 0.0;

  // Backward-compatible per-view RMS errors.
  std::vector<double> per_view_errors;
  std::vector<double> per_view_mean_errors;
  std::vector<double> per_view_p95_errors;
  std::vector<double> per_view_max_errors;

  cv::Mat intrinsic_std_deviations;
  cv::Mat extrinsic_std_deviations;
};

std::vector<cv::Point3f> generateCalibrationObjectPoints(
    const CalibrationBoard& board);

CalibrationDetectionResult detectCalibrationPoints(
    const cv::Mat& image, const CalibrationBoard& board,
    const CircleDetectorParameters& detector_params);

CameraCalibrationResult calibrateCamera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size, const CameraCalibrationOptions& options);

}  // namespace msm3d

#include "msm3d/camera/serialization.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>

namespace msm3d {

bool saveCameraCalibrationResult(
    const std::string& file_path, const CameraCalibrationResult& result,
    const cv::Size& image_size,
    const std::vector<std::string>& valid_view_names) {
  cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "image_width" << image_size.width;
  fs << "image_height" << image_size.height;
  fs << "camera_matrix" << result.camera_matrix;
  fs << "distortion_coefficients" << result.distortion_coefficients;
  fs << "rotation_vectors" << result.rotation_vectors;
  fs << "translation_vectors" << result.translation_vectors;
  fs << "rms" << result.rms;
  fs << "reprojection_error" << result.reprojection_error;
  fs << "mean_reprojection_error" << result.mean_reprojection_error;
  fs << "p95_reprojection_error" << result.p95_reprojection_error;
  fs << "max_reprojection_error" << result.max_reprojection_error;
  fs << "per_view_errors" << result.per_view_errors;
  fs << "per_view_mean_errors" << result.per_view_mean_errors;
  fs << "per_view_p95_errors" << result.per_view_p95_errors;
  fs << "per_view_max_errors" << result.per_view_max_errors;
  fs << "intrinsic_std_deviations" << result.intrinsic_std_deviations;
  fs << "extrinsic_std_deviations" << result.extrinsic_std_deviations;

  fs << "valid_view_names" << "[";
  for (const auto& name : valid_view_names) {
    fs << name;
  }
  fs << "]";

  fs.release();
  return true;
}

CameraCalibrationResult loadCameraCalibrationResult(
    const std::string& result_path) {
  cv::FileStorage fs(result_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open camera calibration result file: " +
                             result_path);
  }

  CameraCalibrationResult result;
  fs["camera_matrix"] >> result.camera_matrix;
  fs["distortion_coefficients"] >> result.distortion_coefficients;
  fs["rms"] >> result.rms;
  fs["reprojection_error"] >> result.reprojection_error;
  if (!fs["mean_reprojection_error"].empty()) {
    fs["mean_reprojection_error"] >> result.mean_reprojection_error;
  }
  if (!fs["p95_reprojection_error"].empty()) {
    fs["p95_reprojection_error"] >> result.p95_reprojection_error;
  }
  if (!fs["max_reprojection_error"].empty()) {
    fs["max_reprojection_error"] >> result.max_reprojection_error;
  }

  const cv::FileNode rvecs_node = fs["rotation_vectors"];
  if (rvecs_node.isSeq()) {
    for (const auto& node : rvecs_node) {
      cv::Mat rvec;
      node >> rvec;
      result.rotation_vectors.push_back(rvec);
    }
  }

  const cv::FileNode tvecs_node = fs["translation_vectors"];
  if (tvecs_node.isSeq()) {
    for (const auto& node : tvecs_node) {
      cv::Mat tvec;
      node >> tvec;
      result.translation_vectors.push_back(tvec);
    }
  }

  if (result.camera_matrix.empty() || result.distortion_coefficients.empty()) {
    throw std::runtime_error(
        "Invalid camera calibration file: missing camera_matrix or "
        "distortion_coefficients.");
  }

  return result;
}

}  // namespace msm3d

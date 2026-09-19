#include "msm3d/camera/calibration.hpp"
#include "msm3d/camera/diagnostics.hpp"
#include "msm3d/camera/serialization.hpp"
#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/camera_calibration.yaml";

  msm3d::CameraCalibrationConfig config;
  try {
    config = msm3d::loadCameraCalibrationConfig(config_path);
    std::cout << "Loaded configuration from: " << config_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load configuration: " << e.what() << std::endl;
    return -1;
  }

  std::vector<std::string> image_paths;
  try {
    image_paths =
        msm3d::DataLoader::loadCameraCalibImagePaths(config.image_dir);
    std::cout << "Found " << image_paths.size()
              << " calibration images in: " << config.image_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load calibration images: " << e.what()
              << std::endl;
    return -1;
  }

  if (image_paths.empty()) {
    std::cerr << "No valid calibration images found in: " << config.image_dir
              << std::endl;
    return -1;
  }

  if (config.save_detection_debug && !config.detection_dir.empty()) {
    std::filesystem::create_directories(config.detection_dir);
  }
  if (config.save_blob_debug && !config.blob_dir.empty()) {
    std::filesystem::create_directories(config.blob_dir);
  }

  const auto object_points_pattern =
      msm3d::generateCalibrationObjectPoints(config.board);
  const cv::Size pattern_size(config.board.columns, config.board.rows);

  std::vector<std::vector<cv::Point3f>> all_object_points;
  std::vector<std::vector<cv::Point2f>> all_image_points;
  std::vector<std::string> valid_view_names;
  cv::Size image_size(0, 0);

  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Failed to read image: " << image_path << std::endl;
      continue;
    }

    if (image_size.width == 0 && image_size.height == 0) {
      image_size = image.size();
    }

    const auto detection = msm3d::detectCalibrationPoints(
        image, config.board, config.circle_detector);
    const std::string filename =
        std::filesystem::path(image_path).filename().string();

    if (detection.found) {
      all_object_points.push_back(object_points_pattern);
      all_image_points.push_back(detection.points);
      valid_view_names.push_back(filename);
      std::cout << "[OK] Pattern detected in: " << filename << std::endl;

      if (config.save_detection_debug && !config.detection_dir.empty()) {
        cv::Mat debug_image = image.clone();
        cv::drawChessboardCorners(debug_image, pattern_size, detection.points,
                                  true);
        cv::imwrite(config.detection_dir + "/" + filename, debug_image);
      }
    } else {
      std::cout << "[FAIL] Failed to detect pattern in: " << filename
                << std::endl;
    }

    if (config.save_blob_debug && !config.blob_dir.empty() &&
        !detection.blob_keypoints.empty()) {
      cv::Mat blob_image;
      cv::drawKeypoints(image, detection.blob_keypoints, blob_image,
                        cv::Scalar(0, 0, 255),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::imwrite(config.blob_dir + "/" + filename, blob_image);
    }
  }

  std::cout << "\nValid views detected: " << all_image_points.size() << " / "
            << image_paths.size() << std::endl;

  if (all_image_points.size() < 3) {
    std::cerr << "Error: Camera calibration requires at least 3 valid views."
              << std::endl;
    return -1;
  }

  std::cout << "Calibration model: fix_k3=" << std::boolalpha
            << config.calibration.fix_k3 << ", zero_tangent_distortion="
            << config.calibration.zero_tangent_distortion
            << ", rational_model=" << config.calibration.use_rational_model
            << std::noboolalpha << std::endl;
  std::cout << "Optimizing camera parameters..." << std::endl;

  const auto calibration = msm3d::calibrateCamera(
      all_object_points, all_image_points, image_size, config.calibration);

  std::cout << "\n--- Calibration Results ---" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Image size: " << image_size.width << " x " << image_size.height
            << std::endl;
  std::cout << "OpenCV overall RMS: " << calibration.rms << " px" << std::endl;
  std::cout << "Computed Euclidean RMS: " << calibration.reprojection_error
            << " px" << std::endl;
  std::cout << "Mean Euclidean reprojection error: "
            << calibration.mean_reprojection_error << " px" << std::endl;
  std::cout << "P95 Euclidean reprojection error: "
            << calibration.p95_reprojection_error << " px" << std::endl;
  std::cout << "Maximum Euclidean reprojection error: "
            << calibration.max_reprojection_error << " px" << std::endl;
  std::cout << "Camera Matrix (K):\n" << calibration.camera_matrix << std::endl;
  std::cout << "Distortion Coefficients (D):\n"
            << calibration.distortion_coefficients << std::endl;

  msm3d::printCameraIntrinsicStdDeviations(
      calibration.intrinsic_std_deviations);
  msm3d::printWorstCameraViews(calibration, valid_view_names, 10);

  const std::filesystem::path result_path(config.result_file);
  if (result_path.has_parent_path()) {
    std::filesystem::create_directories(result_path.parent_path());
  }

  const std::filesystem::path diagnostics_dir =
      (result_path.has_parent_path() ? result_path.parent_path()
                                     : std::filesystem::path(".")) /
      "diagnostics";

  try {
    msm3d::saveCameraResidualDiagnostics(
        diagnostics_dir, all_object_points, all_image_points, valid_view_names,
        calibration, config.board, image_size);
  } catch (const std::exception& e) {
    std::cerr << "Failed to generate residual diagnostics: " << e.what()
              << std::endl;
  }

  if (!msm3d::saveCameraCalibrationResult(config.result_file, calibration,
                                           image_size, valid_view_names)) {
    std::cerr << "Failed to save camera calibration result: "
              << config.result_file << std::endl;
    return -1;
  }

  std::cout << "\nCalibration parameters successfully saved to: "
            << config.result_file << std::endl;
  return 0;
}

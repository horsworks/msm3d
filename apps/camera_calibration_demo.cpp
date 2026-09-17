#include "msm3d/camera_calibration.hpp"
#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/camera_calibration.yaml";

  // 1. 加载配置
  msm3d::CameraCalibrationConfig config;
  try {
    config = msm3d::loadCameraCalibrationConfig(config_path);
    std::cout << "Loaded configuration from: " << config_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load configuration: " << e.what() << std::endl;
    return -1;
  }

  // 2. 利用 DataLoader
  // 统一索引并自然排序标定图像（替代原先手写的目录遍历与排序）
  std::vector<std::string> image_paths;
  try {
    image_paths =
        msm3d::DataLoader::loadCameraCalibImagePaths(config.image_dir);
    std::cout << "Found " << image_paths.size()
              << " calibration images in: " << config.image_dir << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load calibration images: " << e.what() << std::endl;
    return -1;
  }

  if (image_paths.empty()) {
    std::cerr << "No valid calibration images found in: " << config.image_dir
              << std::endl;
    return -1;
  }

  // 3. 创建调试输出目录
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
  cv::Size image_size(0, 0);

  // 4. 逐帧提取特征点
  for (const auto& img_path : image_paths) {
    cv::Mat image = cv::imread(img_path);
    if (image.empty()) {
      std::cerr << "Failed to read image: " << img_path << std::endl;
      continue;
    }

    if (image_size.width == 0 && image_size.height == 0) {
      image_size = image.size();
    }

    const auto result = msm3d::detectCalibrationPoints(image, config.board,
                                                       config.circle_detector);

    const std::string filename =
        std::filesystem::path(img_path).filename().string();

    if (result.found) {
      all_object_points.push_back(object_points_pattern);
      all_image_points.push_back(result.points);
      std::cout << "[OK] Pattern detected in: " << filename << std::endl;

      if (config.save_detection_debug && !config.detection_dir.empty()) {
        cv::Mat debug_img = image.clone();
        cv::drawChessboardCorners(debug_img, pattern_size, result.points, true);
        cv::imwrite(config.detection_dir + "/" + filename, debug_img);
      }
    } else {
      std::cout << "[FAIL] Failed to detect pattern in: " << filename
                << std::endl;
    }

    if (config.save_blob_debug && !config.blob_dir.empty() &&
        !result.blob_keypoints.empty()) {
      cv::Mat blob_img;
      cv::drawKeypoints(image, result.blob_keypoints, blob_img,
                        cv::Scalar(0, 0, 255),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::imwrite(config.blob_dir + "/" + filename, blob_img);
    }
  }

  std::cout << "\nValid views detected: " << all_image_points.size() << " / "
            << image_paths.size() << std::endl;

  if (all_image_points.size() < 3) {
    std::cerr << "Error: Camera calibration requires at least 3 valid views."
              << std::endl;
    return -1;
  }

  // 5. 标定解算
  std::cout << "Optimizing camera parameters..." << std::endl;
  const auto calib_result = msm3d::calibrateCamera(
      all_object_points, all_image_points, image_size, config.calibration);

  std::cout << "\n--- Calibration Results ---" << std::endl;
  std::cout << "Image size: " << image_size.width << " x " << image_size.height
            << std::endl;
  std::cout << "Overall RMS error: " << calib_result.rms << std::endl;
  std::cout << "Mean reprojection error: " << calib_result.reprojection_error
            << " px" << std::endl;
  std::cout << "Camera Matrix (K):\n"
            << calib_result.camera_matrix << std::endl;
  std::cout << "Distortion Coefficients (D):\n"
            << calib_result.distortion_coefficients << std::endl;

  // 6. 保存标定结果
  std::filesystem::path res_path(config.result_file);
  if (res_path.has_parent_path()) {
    std::filesystem::create_directories(res_path.parent_path());
  }

  cv::FileStorage fs(config.result_file, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open result file for writing: "
              << config.result_file << std::endl;
    return -1;
  }

  fs << "image_width" << image_size.width;
  fs << "image_height" << image_size.height;
  fs << "camera_matrix" << calib_result.camera_matrix;
  fs << "distortion_coefficients" << calib_result.distortion_coefficients;
  fs << "rotation_vectors" << calib_result.rotation_vectors;
  fs << "translation_vectors" << calib_result.translation_vectors;
  fs << "rms" << calib_result.rms;
  fs << "reprojection_error" << calib_result.reprojection_error;
  fs << "per_view_errors" << calib_result.per_view_errors;
  fs.release();

  std::cout << "Calibration parameters successfully saved to: "
            << config.result_file << std::endl;

  return 0;
}
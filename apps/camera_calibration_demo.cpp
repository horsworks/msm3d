#include "msm3d/camera_calibration.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

static std::vector<std::string> CollectImages(const std::string& folder) {
  std::vector<std::string> files;

  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const auto path = entry.path().string();

    if (path.find(".png") != std::string::npos ||
        path.find(".jpg") != std::string::npos ||
        path.find(".bmp") != std::string::npos) {
      files.push_back(path);
    }
  }

  std::sort(files.begin(), files.end());

  return files;
}

static void SaveDetectionImage(
    const std::string& filename, const cv::Mat& image,
    const msm3d::CalibrationDetectionResult& detection,
    const std::string& output_dir) {
  cv::Mat debug;

  if (image.channels() == 1) {
    cv::cvtColor(image, debug, cv::COLOR_GRAY2BGR);
  } else {
    debug = image.clone();
  }

  if (detection.found) {
    cv::drawChessboardCorners(
        debug, cv::Size(static_cast<int>(detection.points.size()), 1),
        detection.points, true);

    cv::drawKeypoints(debug, detection.blob_keypoints, debug,
                      cv::Scalar(0, 255, 0));
  }

  const std::filesystem::path path(filename);

  const std::string output =
      output_dir + "/" + path.stem().string() + "_detect.png";

  cv::imwrite(output, debug);
}

int main() {
  using namespace msm3d;

  CameraCalibrationConfig config =
      LoadCameraCalibrationConfig("./config/camera_calibration.yaml");

  std::filesystem::create_directories(config.detection_dir);

  std::filesystem::create_directories(config.blob_dir);

  const auto image_files = CollectImages(config.image_dir);

  std::vector<std::vector<cv::Point3f>> object_points;

  std::vector<std::vector<cv::Point2f>> image_points;

  const auto board_points = GenerateCalibrationObjectPoints(config.board);

  cv::Size image_size;

  for (const auto& file : image_files) {
    cv::Mat image = cv::imread(file);

    if (image.empty()) {
      continue;
    }

    image_size = image.size();

    auto detection =
        DetectCalibrationPoints(image, config.board, config.circle_detector);

    if (config.save_detection_debug) {
      SaveDetectionImage(file, image, detection, config.detection_dir);
    }

    if (!detection.found) {
      std::cout << "Detection failed: " << file << std::endl;

      continue;
    }

    object_points.push_back(board_points);

    image_points.push_back(detection.points);

    std::cout << "Detection success: " << file << std::endl;
  }

  auto result = CalibrateCamera(object_points, image_points, image_size,
                                config.calibration);

  std::cout << "RMS: " << result.rms << std::endl;

  std::cout << "Reprojection error: " << result.reprojection_error << std::endl;

  cv::FileStorage fs(config.result_file, cv::FileStorage::WRITE);

  fs << "camera_matrix" << result.camera_matrix;

  fs << "distortion_coefficients" << result.distortion_coefficients;

  fs << "rms" << result.rms;

  fs << "reprojection_error" << result.reprojection_error;

  fs.release();

  return 0;
}

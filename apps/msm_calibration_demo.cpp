#include "msm3d/io/config.hpp"
#include "msm3d/msm_calibration.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/msm_calibration.yaml";

  // 1. 读取标定配置
  msm3d::MsmCalibrationConfig config;
  try {
    config = msm3d::loadMsmCalibrationConfig(config_path);
    std::cout << "Loaded MSM config from: " << config_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Config load error: " << e.what() << std::endl;
    return -1;
  }

  // 2. 读取相机标定先验
  msm3d::CameraCalibrationResult camera_calib;
  try {
    camera_calib = msm3d::loadCameraCalibrationResult(config.camera_param_file);
    std::cout << "Loaded camera calibration from: " << config.camera_param_file
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Camera param load error: " << e.what() << std::endl;
    return -1;
  }

  // 3. 读取各视角绝对相位图 (支持 pose_<id>.exr 及 <id>.exr)
  const int total_poses =
      static_cast<int>(camera_calib.rotation_vectors.size());
  std::vector<cv::Mat> all_phase_maps(total_poses);

  std::cout << "Loading phase maps from: " << config.phase_folder << " ..."
            << std::endl;

  int loaded_count = 0;
  for (int pid = 0; pid < total_poses; ++pid) {
    std::string path =
        config.phase_folder + "/pose_" + std::to_string(pid) + ".exr";

    if (!std::filesystem::exists(path)) {
      path = config.phase_folder + "/" + std::to_string(pid) + ".exr";
    }

    if (std::filesystem::exists(path)) {
      all_phase_maps[pid] = cv::imread(path, cv::IMREAD_UNCHANGED);

      if (!all_phase_maps[pid].empty()) {
        ++loaded_count;
      }
    }
  }

  std::cout << "Successfully loaded " << loaded_count << " / " << total_poses
            << " phase maps." << std::endl;

  if (loaded_count == 0) {
    std::cerr << "Error: No valid phase maps loaded." << std::endl;
    return -1;
  }

  // 4. 执行振镜系统标定与模型解算
  std::cout << "\nStarting MSM calibration pipeline..." << std::endl;

  msm3d::MsmCalibrationResult result;
  try {
    result = msm3d::calibrateMsm(config, camera_calib, all_phase_maps);
  } catch (const std::exception& e) {
    std::cerr << "MSM calibration failed: " << e.what() << std::endl;
    return -1;
  }

  // 5. 保存标定结果
  if (!config.result_file.empty()) {
    std::filesystem::path result_path(config.result_file);

    if (result_path.has_parent_path()) {
      std::filesystem::create_directories(result_path.parent_path());
    }

    if (msm3d::saveMsmCalibrationResult(config.result_file, result)) {
      std::cout << "Saved MSM calibration parameters to: " << config.result_file
                << std::endl;
    } else {
      std::cerr << "Failed to save result to: " << config.result_file
                << std::endl;
    }
  }

  return 0;
}

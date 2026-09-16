#pragma once

#include "msm3d/camera_calibration.hpp"

#include <string>
#include <vector>

namespace msm3d {

// 通用条纹参数（供振镜标定、物体重建等后续所有阶段复用）
struct FringePatternConfig {
  int steps = 24;
  std::vector<int> frequencies;
  std::string extension = ".bmp";
};

// 相位计算 / 多位姿条纹分析配置
struct PhaseConfig {
  std::string fringe_folder;
  int pose_count = 0;
  FringePatternConfig pattern;
  std::string output_folder;
};

// 统一的配置反序列化函数
CameraCalibrationConfig loadCameraCalibrationConfig(
    const std::string& config_path);

PhaseConfig loadPhaseConfig(const std::string& config_path);

}  // namespace msm3d
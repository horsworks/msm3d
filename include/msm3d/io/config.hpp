#pragma once

#include "msm3d/camera_calibration.hpp"
#include "msm3d/msm_calibration.hpp"

#include <string>
#include <vector>

namespace msm3d {

struct FringePatternConfig {
  int steps = 24;
  std::vector<int> frequencies;
  std::string extension = ".bmp";
};

struct PhaseConfig {
  std::string fringe_folder;
  int pose_count = 0;
  FringePatternConfig pattern;
  std::string output_folder;
  double min_modulation = 8.0;
  int median_filter_size = 3;
};

CameraCalibrationConfig loadCameraCalibrationConfig(
    const std::string& config_path);

CameraCalibrationResult loadCameraCalibrationResult(
    const std::string& result_path);

PhaseConfig loadPhaseConfig(const std::string& config_path);

MsmCalibrationConfig loadMsmCalibrationConfig(const std::string& config_path);

}  // namespace msm3d
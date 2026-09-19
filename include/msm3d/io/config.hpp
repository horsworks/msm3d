#pragma once

#include "msm3d/camera/types.hpp"
#include "msm3d/msm/types.hpp"
#include "msm3d/phase/quality.hpp"

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
  PhaseQualityOptions quality;

  // "none" or "median".
  std::string phase_filter = "median";
  int median_filter_size = 3;
  bool save_confidence_map = true;
};

CameraCalibrationConfig loadCameraCalibrationConfig(
    const std::string& config_path);

PhaseConfig loadPhaseConfig(const std::string& config_path);

MsmCalibrationConfig loadMsmCalibrationConfig(const std::string& config_path);

}  // namespace msm3d

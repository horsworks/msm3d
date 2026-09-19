#pragma once

#include "msm3d/camera_calibration.hpp"
#include "msm3d/msm_calibration.hpp"
#include "msm3d/phase_processor.hpp"

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

  // "none", "median", or "local_plane".
  std::string phase_filter = "local_plane";
  int median_filter_size = 3;
  int local_plane_filter_size = 3;
  int local_plane_min_valid_neighbors = 5;
  double local_plane_robust_scale_rad = 0.10;

  bool save_confidence_map = true;
};

CameraCalibrationConfig loadCameraCalibrationConfig(
    const std::string& config_path);

CameraCalibrationResult loadCameraCalibrationResult(
    const std::string& result_path);

PhaseConfig loadPhaseConfig(const std::string& config_path);

MsmCalibrationConfig loadMsmCalibrationConfig(const std::string& config_path);

}  // namespace msm3d
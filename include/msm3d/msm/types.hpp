#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

struct BoardPlane {
  cv::Vec3d normal;
  double d = 0.0;
};

struct DiscretePlane {
  double psi = 0.0;
  cv::Vec3d normal;
  double d = 0.0;

  double rms_mm = 0.0;
  double spread_ratio = 0.0;
  double thickness_ratio = 0.0;
  double inlier_ratio = 0.0;
  double confidence = 1.0;

  int point_count = 0;
  int inlier_count = 0;
  int pose_count = 0;
  bool valid = false;
};

struct MsmCalibrationOptions {
  bool auto_phase_range = true;
  int min_covisible_poses = 3;
  double min_psi = 25.0;
  double max_psi = 135.0;

  int plane_count = 30;
  int harmonic_order = 2;

  int angle_resampling_iterations = 3;
  double angle_resampling_tolerance = 0.01;

  int iso_fit_half_window = 2;
  double min_local_phase_gradient = 0.01;
  double max_local_phase_gradient = 0.5;
  double max_lateral_jump_px = 5.0;

  int min_points_per_pose = 30;

  double plane_inlier_threshold_mm = 0.8;
  double min_spread_ratio = 0.02;
  double max_thickness_ratio = 0.1;
  double max_plane_rms_mm = 1.5;
  double thickness_soft_scale = 0.02;
  double plane_confidence_rms_floor_mm = 0.05;
  double harmonic_regularization = 1e-3;

  int diagnostic_phase_bins = 10;
};

struct MsmCalibrationConfig {
  std::string camera_param_file;
  std::string phase_folder;
  std::vector<int> train_poses;
  std::vector<int> test_poses;
  MsmCalibrationOptions options;
  std::string result_file;
};

}  // namespace msm3d

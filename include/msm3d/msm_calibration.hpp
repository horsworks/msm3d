#pragma once

#include "msm3d/camera_calibration.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
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
  double spread_ratio = 0.0;     // sigma2 / sigma1
  double thickness_ratio = 0.0;  // sigma3 / sigma2
  double inlier_ratio = 0.0;

  // Plane-level confidence. The main pipeline normalizes valid-plane
  // confidences to median ~= 1 before geometry solving.
  double confidence = 1.0;

  // Backward-compatible alias used by the current downstream code.
  double weight = 1.0;

  int point_count = 0;
  int inlier_count = 0;
  int pose_count = 0;
  bool valid = false;
};

struct RationalAngleModel {
  double psi_ref = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;

  // Low-degree residual correction on top of the rational trend.
  // Runtime evaluation uses the same piecewise-linear basis used for fitting.
  std::vector<double> correction_psi;
  std::vector<double> correction_alpha;

  bool valid = false;

  double evaluateBase(double psi) const;
  double evaluateCorrection(double psi) const;
  double evaluate(double psi) const;
};

struct HarmonicDriftModel {
  int order = 1;

  // Anchored harmonic basis:
  // [cos(theta)-1, sin(theta), cos(2theta)-1, sin(2theta), ...]
  // The fixed offset is absorbed into nominal_center_s0, so the harmonic
  // correction is exactly zero at theta = 0.
  std::vector<double> beta_u;
  std::vector<double> beta_v;
  bool valid = false;

  void evaluate(double theta_rad, double& out_delta_u,
                double& out_delta_v) const;
};

struct MsmCalibrationOptions {
  // Phase range / visibility.
  bool auto_phase_range = true;
  int min_covisible_poses = 3;
  double min_psi = 25.0;
  double max_psi = 135.0;

  // Discrete light-plane sampling.
  int plane_count = 30;
  int harmonic_order = 2;

  // Repeat uniform-optical-angle resampling until the phase samples converge.
  int angle_resampling_iterations = 3;
  double angle_resampling_tolerance = 0.01;

  // Fit a small smooth residual correction after the final rational-model
  // resampling has converged. The correction never participates in resampling.
  bool angle_correction_enabled = true;
  int angle_correction_knots = 9;
  double angle_correction_smoothness = 10.0;
  double angle_correction_max_abs_mrad = 2.5;

  // Local linear subpixel extraction.
  int iso_fit_half_window = 2;
  double min_local_phase_gradient = 0.01;
  double max_local_phase_gradient = 0.5;
  double max_lateral_jump_px = 5.0;

  // Per-pose observation validity.
  int min_points_per_pose = 30;

  // Robust plane fitting / geometry quality.
  double plane_inlier_threshold_mm = 0.8;
  double min_spread_ratio = 0.02;
  double max_thickness_ratio = 0.1;
  double max_plane_rms_mm = 1.5;

  // Thickness is primarily a soft confidence term. max_thickness_ratio remains
  // a broad sanity limit for clearly non-planar observations.
  double thickness_soft_scale = 0.02;

  // Prevent unrealistically tiny RMS from producing extreme confidence.
  double plane_confidence_rms_floor_mm = 0.05;

  // Tikhonov regularization for harmonic coefficients. The reference-center
  // correction itself is left unregularized.
  double harmonic_regularization = 1e-3;

  // Test-pose reconstruction diagnostics across phase. 0 disables bin output.
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

struct MsmCalibrationResult {
  cv::Vec3d nominal_axis_w;
  cv::Vec3d nominal_center_s0;
  cv::Vec3d ref_normal_n0;
  cv::Vec3d basis_u;
  cv::Vec3d basis_v;

  RationalAngleModel angle_model;
  HarmonicDriftModel harmonic_drift;

  double train_rmse_mm = 0.0;
  double test_rmse_mm = 0.0;

  void evaluatePlane(double psi, cv::Vec3d& out_normal, double& out_d) const;
};

std::pair<double, double> autoDetectValidPhaseRange(
    const std::vector<cv::Mat>& phase_maps, int min_covisible_poses = 3);

BoardPlane computeBoardPlane(const cv::Mat& rvec_or_R, const cv::Mat& t_mm);

// Local linear fit:
//   psi(x) ~= a*x + b
// around a detected target-phase crossing.
// Existing call sites remain valid because the new parameters have defaults.
std::vector<cv::Point2d> extractIsoPhaseSubpixels(
    const cv::Mat& phase_map, double target_psi,
    const cv::Mat& mask = cv::Mat(), double grad_threshold = 1e-3,
    int fit_half_window = 2, double max_grad_threshold = 0.5,
    double max_lateral_jump_px = 5.0);

std::vector<cv::Vec3d> projectSubpixelsToBoard(
    const std::vector<cv::Point2d>& subpixels, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const BoardPlane& board_plane,
    double min_depth_mm = 50.0, double max_depth_mm = 2000.0);

// Points are grouped by pose. Each pose has equal
// total base weight, so a longer iso-phase curve does not dominate the plane.
DiscretePlane fitPlaneRobustTLS(
    const std::vector<std::vector<cv::Vec3d>>& points_by_pose, double psi,
    const MsmCalibrationOptions& options);

// Backward-compatible overload. It treats the input as one observation group.
DiscretePlane fitPlaneRobustTLS(const std::vector<cv::Vec3d>& points,
                                double psi,
                                const MsmCalibrationOptions& options);

void enforceNormalConsistency(std::vector<DiscretePlane>& planes);

bool solveNominalRotationGeometry(const std::vector<DiscretePlane>& planes,
                                  double ref_psi, cv::Vec3d& out_w,
                                  cv::Vec3d& out_S0, cv::Vec3d& out_n0,
                                  cv::Vec3d& out_u, cv::Vec3d& out_v);

std::vector<double> computeRelativeAngles(
    const std::vector<DiscretePlane>& planes, const cv::Vec3d& w,
    const cv::Vec3d& n0, bool enforce_monotonic = true);

RationalAngleModel fitRationalAngleModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    double psi_ref);

HarmonicDriftModel fitHarmonicDriftModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    const cv::Vec3d& w, const cv::Vec3d& initial_s0, const cv::Vec3d& u,
    const cv::Vec3d& v, cv::Vec3d& out_reference_s0, int order = 1,
    double regularization = 1e-3);

double evaluateMsmReconstruction(const MsmCalibrationResult& result,
                                 const cv::Mat& phase_map,
                                 const cv::Mat& camera_matrix,
                                 const cv::Mat& dist_coeffs,
                                 const BoardPlane& board_plane, double min_psi,
                                 double max_psi, int pixel_stride = 4);

MsmCalibrationResult calibrateMsm(const MsmCalibrationConfig& config,
                                  const CameraCalibrationResult& camera_calib,
                                  const std::vector<cv::Mat>& all_phase_maps);

bool saveMsmCalibrationResult(const std::string& file_path,
                              const MsmCalibrationResult& result);

MsmCalibrationResult loadMsmCalibrationResult(const std::string& file_path);

}  // namespace msm3d

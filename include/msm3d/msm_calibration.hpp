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
  double weight = 1.0;
  int point_count = 0;
  bool valid = false;
};

struct RationalAngleModel {
  double psi_ref = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  bool valid = false;

  double evaluate(double psi) const;
};

struct HarmonicDriftModel {
  int order = 1;
  std::vector<double> beta_u;
  std::vector<double> beta_v;
  bool valid = false;

  void evaluate(double theta_rad, double& out_delta_u,
                double& out_delta_v) const;
};

struct MsmCalibrationOptions {
  // 自适应多位姿共见区间配置
  bool auto_phase_range = true;  // 是否自动计算重叠共见相位区间
  int min_covisible_poses = 3;   // 单个等相位面要求的最少共见位姿数
  double min_psi = 25.0;         // auto_phase_range=false 时的保底参数
  double max_psi = 135.0;
  int plane_count = 30;    // 离散光平面采样数
  int harmonic_order = 2;  // 谐波阶数
  double min_spread_ratio = 0.02;
  double max_thickness_ratio = 5e-3;
  double max_plane_rms_mm = 1.5;
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

// 自动计算多位姿充分共见的有效绝对相位区间
std::pair<double, double> autoDetectValidPhaseRange(
    const std::vector<cv::Mat>& phase_maps, int min_covisible_poses = 3);

BoardPlane computeBoardPlane(const cv::Mat& rvec_or_R, const cv::Mat& t_mm);

std::vector<cv::Point2d> extractIsoPhaseSubpixels(
    const cv::Mat& phase_map, double target_psi,
    const cv::Mat& mask = cv::Mat(), double grad_threshold = 1e-3);

std::vector<cv::Vec3d> projectSubpixelsToBoard(
    const std::vector<cv::Point2d>& subpixels, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const BoardPlane& board_plane,
    double min_depth_mm = 50.0, double max_depth_mm = 2000.0);

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
    const cv::Vec3d& w, const cv::Vec3d& S0, const cv::Vec3d& u,
    const cv::Vec3d& v, int order = 1);

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
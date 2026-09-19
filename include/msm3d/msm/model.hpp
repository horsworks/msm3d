#pragma once

#include "msm3d/msm/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace msm3d {

struct RationalAngleModel {
  double psi_ref = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  bool valid = false;

  double evaluateBase(double psi) const;
  double evaluate(double psi) const;
};

struct HarmonicDriftModel {
  int order = 1;

  // Anchored basis:
  // [cos(theta)-1, sin(theta), cos(2theta)-1, sin(2theta), ...].
  std::vector<double> beta_u;
  std::vector<double> beta_v;
  bool valid = false;

  void evaluate(double theta_rad, double& out_delta_u,
                double& out_delta_v) const;
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

}  // namespace msm3d

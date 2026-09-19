#include "msm3d/msm/model.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>

namespace msm3d {

double RationalAngleModel::evaluateBase(double psi) const {
  if (!valid) {
    return 0.0;
  }

  const double delta_psi = psi - psi_ref;
  const double denominator = b0 + b1 * delta_psi;

  if (!std::isfinite(denominator) || std::abs(denominator) < 1e-12) {
    return 0.0;
  }

  return std::atan2(delta_psi, denominator);
}

double RationalAngleModel::evaluate(double psi) const {
  return evaluateBase(psi);
}

void HarmonicDriftModel::evaluate(double theta_rad, double& out_delta_u,
                                  double& out_delta_v) const {
  out_delta_u = 0.0;
  out_delta_v = 0.0;

  if (!valid || order <= 0) {
    return;
  }

  for (int k = 1; k <= order; ++k) {
    const int idx = 2 * (k - 1);
    const double ang = static_cast<double>(k) * theta_rad;
    const double cos_basis = std::cos(ang) - 1.0;
    const double sin_basis = std::sin(ang);

    if (idx + 1 < static_cast<int>(beta_u.size())) {
      out_delta_u += beta_u[idx] * cos_basis + beta_u[idx + 1] * sin_basis;
    }
    if (idx + 1 < static_cast<int>(beta_v.size())) {
      out_delta_v += beta_v[idx] * cos_basis + beta_v[idx + 1] * sin_basis;
    }
  }
}

void MsmCalibrationResult::evaluatePlane(double psi, cv::Vec3d& out_normal,
                                         double& out_d) const {
  const double theta = angle_model.evaluate(psi);

  cv::Mat R_theta;
  const cv::Vec3d rvec = nominal_axis_w * theta;
  cv::Rodrigues(rvec, R_theta);

  const cv::Mat n0_mat = (cv::Mat_<double>(3, 1) << ref_normal_n0[0],
                          ref_normal_n0[1], ref_normal_n0[2]);
  const cv::Mat n_mat = R_theta * n0_mat;

  out_normal = cv::normalize(
      cv::Vec3d(n_mat.at<double>(0), n_mat.at<double>(1), n_mat.at<double>(2)));

  double delta_u = 0.0;
  double delta_v = 0.0;
  if (harmonic_drift.valid && harmonic_drift.order > 0) {
    harmonic_drift.evaluate(theta, delta_u, delta_v);
  }

  const cv::Vec3d S = nominal_center_s0 + delta_u * basis_u + delta_v * basis_v;
  out_d = -out_normal.dot(S);
}

}  // namespace msm3d

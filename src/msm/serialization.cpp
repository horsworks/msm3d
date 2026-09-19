#include "msm3d/msm/serialization.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace msm3d {

bool saveMsmCalibrationResult(const std::string& file_path,
                              const MsmCalibrationResult& result) {
  cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "nominal_axis_w" << cv::Mat(result.nominal_axis_w);
  fs << "nominal_center_s0" << cv::Mat(result.nominal_center_s0);
  fs << "ref_normal_n0" << cv::Mat(result.ref_normal_n0);
  fs << "basis_u" << cv::Mat(result.basis_u);
  fs << "basis_v" << cv::Mat(result.basis_v);

  fs << "psi_ref" << result.angle_model.psi_ref;
  fs << "angle_model_b0" << result.angle_model.b0;
  fs << "angle_model_b1" << result.angle_model.b1;

  fs << "harmonic_order" << result.harmonic_drift.order;
  fs << "harmonic_basis" << "anchored_cos_minus_one";
  fs << "beta_u" << result.harmonic_drift.beta_u;
  fs << "beta_v" << result.harmonic_drift.beta_v;

  fs << "train_rmse_mm" << result.train_rmse_mm;
  fs << "test_rmse_mm" << result.test_rmse_mm;

  fs.release();
  return true;
}

MsmCalibrationResult loadMsmCalibrationResult(const std::string& file_path) {
  cv::FileStorage fs(file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Cannot open MSM result file: " + file_path);
  }

  MsmCalibrationResult result;
  cv::Mat w_mat;
  cv::Mat s0_mat;
  cv::Mat n0_mat;
  cv::Mat u_mat;
  cv::Mat v_mat;

  fs["nominal_axis_w"] >> w_mat;
  fs["nominal_center_s0"] >> s0_mat;
  fs["ref_normal_n0"] >> n0_mat;
  fs["basis_u"] >> u_mat;
  fs["basis_v"] >> v_mat;

  result.nominal_axis_w = cv::Vec3d(w_mat);
  result.nominal_center_s0 = cv::Vec3d(s0_mat);
  result.ref_normal_n0 = cv::Vec3d(n0_mat);
  result.basis_u = cv::Vec3d(u_mat);
  result.basis_v = cv::Vec3d(v_mat);

  fs["psi_ref"] >> result.angle_model.psi_ref;

  const cv::FileNode b0_node = fs["angle_model_b0"];
  const cv::FileNode b1_node = fs["angle_model_b1"];

  if (!b0_node.empty() && !b1_node.empty()) {
    b0_node >> result.angle_model.b0;
    b1_node >> result.angle_model.b1;
  } else {
    double old_a1 = 0.0;
    double old_a2 = 0.0;
    fs["angle_model_a1"] >> old_a1;
    fs["angle_model_a2"] >> old_a2;

    result.angle_model.b1 = old_a1;
    result.angle_model.b0 = old_a1 * result.angle_model.psi_ref + old_a2;
  }

  result.angle_model.valid = true;

  fs["harmonic_order"] >> result.harmonic_drift.order;

  std::vector<double> stored_beta_u;
  std::vector<double> stored_beta_v;
  fs["beta_u"] >> stored_beta_u;
  fs["beta_v"] >> stored_beta_v;

  std::string harmonic_basis;
  const cv::FileNode basis_node = fs["harmonic_basis"];
  if (!basis_node.empty()) {
    basis_node >> harmonic_basis;
  }

  const int order = result.harmonic_drift.order;
  const std::size_t anchored_size =
      static_cast<std::size_t>(std::max(0, 2 * order));
  const std::size_t legacy_size =
      static_cast<std::size_t>(std::max(0, 2 * order + 1));

  if (harmonic_basis == "anchored_cos_minus_one" &&
      stored_beta_u.size() == anchored_size &&
      stored_beta_v.size() == anchored_size) {
    result.harmonic_drift.beta_u = std::move(stored_beta_u);
    result.harmonic_drift.beta_v = std::move(stored_beta_v);
    result.harmonic_drift.valid = (order > 0);
  } else if (order > 0 && stored_beta_u.size() == legacy_size &&
             stored_beta_v.size() == legacy_size) {
    // Legacy representation:
    // delta = dc + sum(A_k cos(k*theta) + B_k sin(k*theta)).
    // Move delta(0) into the reference center and convert the remaining
    // coefficients to A_k*(cos(k*theta)-1) + B_k*sin(k*theta).
    double delta_u_at_zero = stored_beta_u[0];
    double delta_v_at_zero = stored_beta_v[0];

    result.harmonic_drift.beta_u.assign(anchored_size, 0.0);
    result.harmonic_drift.beta_v.assign(anchored_size, 0.0);

    for (int k = 1; k <= order; ++k) {
      const int legacy_idx = 1 + 2 * (k - 1);
      const int anchored_idx = 2 * (k - 1);

      const double au = stored_beta_u[legacy_idx];
      const double bu = stored_beta_u[legacy_idx + 1];
      const double av = stored_beta_v[legacy_idx];
      const double bv = stored_beta_v[legacy_idx + 1];

      delta_u_at_zero += au;
      delta_v_at_zero += av;

      result.harmonic_drift.beta_u[anchored_idx] = au;
      result.harmonic_drift.beta_u[anchored_idx + 1] = bu;
      result.harmonic_drift.beta_v[anchored_idx] = av;
      result.harmonic_drift.beta_v[anchored_idx + 1] = bv;
    }

    result.nominal_center_s0 +=
        delta_u_at_zero * result.basis_u + delta_v_at_zero * result.basis_v;
    result.harmonic_drift.valid = true;
  } else {
    result.harmonic_drift.beta_u.clear();
    result.harmonic_drift.beta_v.clear();
    result.harmonic_drift.valid = false;
  }

  fs["train_rmse_mm"] >> result.train_rmse_mm;
  fs["test_rmse_mm"] >> result.test_rmse_mm;

  fs.release();
  return result;
}

}  // namespace msm3d

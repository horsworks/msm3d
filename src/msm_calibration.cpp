#include "msm3d/msm_calibration.hpp"
#include "msm3d/camera_calibration.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace msm3d {

double RationalAngleModel::evaluate(double psi) const {
  if (!valid) return 0.0;
  const double num = psi - psi_ref;
  const double den = a1 * psi + a2;
  if (std::abs(den) < 1e-9) return 0.0;
  return std::atan(num / den);
}

// 统一解析：常数项位于 beta_u[0] / beta_v[0]，后续为 cos/sin 谐波项
void HarmonicDriftModel::evaluate(double theta_rad, double& out_delta_u,
                                  double& out_delta_v) const {
  out_delta_u = 0.0;
  out_delta_v = 0.0;
  if (!valid || order <= 0 || beta_u.empty() || beta_v.empty()) return;

  out_delta_u = beta_u[0];
  out_delta_v = beta_v[0];

  for (int k = 1; k <= order; ++k) {
    const int idx_cos = 1 + 2 * (k - 1);
    const int idx_sin = 1 + 2 * (k - 1) + 1;
    const double ang = static_cast<double>(k) * theta_rad;
    const double c = std::cos(ang);
    const double s = std::sin(ang);

    if (idx_sin < static_cast<int>(beta_u.size())) {
      out_delta_u += beta_u[idx_cos] * c + beta_u[idx_sin] * s;
    }
    if (idx_sin < static_cast<int>(beta_v.size())) {
      out_delta_v += beta_v[idx_cos] * c + beta_v[idx_sin] * s;
    }
  }
}

void MsmCalibrationResult::evaluatePlane(double psi, cv::Vec3d& out_normal,
                                         double& out_d) const {
  const double theta = angle_model.evaluate(psi);

  // 1. 名义法向量沿转轴 w 旋转 theta
  cv::Mat R_theta;
  const cv::Vec3d rvec = nominal_axis_w * theta;
  cv::Rodrigues(rvec, R_theta);

  const cv::Mat n0_mat = (cv::Mat_<double>(3, 1) << ref_normal_n0[0],
                          ref_normal_n0[1], ref_normal_n0[2]);
  const cv::Mat n_mat = R_theta * n0_mat;

  out_normal = cv::normalize(
      cv::Vec3d(n_mat.at<double>(0), n_mat.at<double>(1), n_mat.at<double>(2)));

  // 2. 轴心漂移补偿 (包含常数项更新与动态摆动)
  double delta_u = 0.0, delta_v = 0.0;
  if (harmonic_drift.valid && harmonic_drift.order > 0) {
    harmonic_drift.evaluate(theta, delta_u, delta_v);
  }

  const cv::Vec3d S = nominal_center_s0 + delta_u * basis_u + delta_v * basis_v;
  out_d = -out_normal.dot(S);
}

BoardPlane computeBoardPlane(const cv::Mat& rvec_or_R, const cv::Mat& t_mm) {
  cv::Mat R_64, t_64;
  rvec_or_R.convertTo(R_64, CV_64F);
  t_mm.convertTo(t_64, CV_64F);

  cv::Mat R;
  if (R_64.rows == 3 && R_64.cols == 3) {
    R = R_64;
  } else {
    cv::Rodrigues(R_64, R);
  }

  cv::Vec3d n(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
  n = cv::normalize(n);
  if (n[2] < 0.0) {
    n = -n;
  }

  const double* t_ptr = t_64.ptr<double>(0);
  cv::Vec3d t(t_ptr[0], t_ptr[1], t_ptr[2]);

  if (cv::norm(t) > 1e-4 && cv::norm(t) < 5.0) {
    t *= 1000.0;
  }

  const double d = -n.dot(t);
  return {n, d};
}

std::pair<double, double> autoDetectValidPhaseRange(
    const std::vector<cv::Mat>& phase_maps, int min_covisible_poses) {
  if (phase_maps.empty()) return {0.0, 0.0};

  struct PoseInterval {
    double p_low = 0.0;
    double p_high = 0.0;
  };

  std::vector<PoseInterval> intervals;
  intervals.reserve(phase_maps.size());

  for (const auto& pmap : phase_maps) {
    if (pmap.empty()) continue;

    cv::Mat p64;
    if (pmap.channels() > 1) {
      cv::extractChannel(pmap, p64, 0);
    } else {
      p64 = pmap;
    }
    p64.convertTo(p64, CV_64F);

    std::vector<double> valid_vals;
    valid_vals.reserve(p64.rows * p64.cols / 16);

    for (int y = 0; y < p64.rows; y += 4) {
      const double* r_ptr = p64.ptr<double>(y);
      for (int x = 0; x < p64.cols; x += 4) {
        const double v = r_ptr[x];
        if (std::isfinite(v)) {
          valid_vals.push_back(v);
        }
      }
    }

    if (valid_vals.size() < 2000) continue;

    std::sort(valid_vals.begin(), valid_vals.end());
    const std::size_t idx_03 =
        static_cast<std::size_t>(valid_vals.size() * 0.03);
    const std::size_t idx_97 =
        static_cast<std::size_t>(valid_vals.size() * 0.97);

    intervals.push_back({valid_vals[idx_03], valid_vals[idx_97]});
  }

  if (intervals.empty()) return {0.0, 0.0};

  double global_min = 1e9, global_max = -1e9;
  for (const auto& inv : intervals) {
    global_min = std::min(global_min, inv.p_low);
    global_max = std::max(global_max, inv.p_high);
  }

  const int required_poses = std::max(
      2, std::min(min_covisible_poses, static_cast<int>(intervals.size())));
  const double step = 0.5;
  double best_start = 0.0, best_end = 0.0;
  double cur_start = -1.0;
  double max_len = 0.0;

  for (double psi = global_min; psi <= global_max; psi += step) {
    int covisible_count = 0;
    for (const auto& inv : intervals) {
      if (psi >= inv.p_low && psi <= inv.p_high) {
        covisible_count++;
      }
    }

    if (covisible_count >= required_poses) {
      if (cur_start < 0.0) cur_start = psi;
    } else {
      if (cur_start >= 0.0) {
        const double len = (psi - step) - cur_start;
        if (len > max_len) {
          max_len = len;
          best_start = cur_start;
          best_end = psi - step;
        }
        cur_start = -1.0;
      }
    }
  }

  if (cur_start >= 0.0) {
    const double len = global_max - cur_start;
    if (len > max_len) {
      best_start = cur_start;
      best_end = global_max;
    }
  }

  best_start += 2.0;
  best_end -= 2.0;
  if (best_end <= best_start) return {global_min, global_max};

  return {best_start, best_end};
}

std::vector<cv::Point2d> extractIsoPhaseSubpixels(const cv::Mat& phase_map,
                                                  double target_psi,
                                                  const cv::Mat& mask,
                                                  double grad_threshold) {
  if (phase_map.empty()) return {};

  cv::Mat phase_f64;
  if (phase_map.channels() > 1) {
    cv::extractChannel(phase_map, phase_f64, 0);
  } else {
    phase_f64 = phase_map;
  }
  if (phase_f64.type() != CV_64F) {
    phase_f64.convertTo(phase_f64, CV_64F);
  }

  const int rows = phase_f64.rows;
  const int cols = phase_f64.cols;
  const bool has_mask = (!mask.empty() && mask.size() == phase_f64.size());

  const double min_phys_grad = std::max(grad_threshold, 0.01);
  const double max_phys_grad = 0.5;

  std::vector<cv::Point2d> raw_subpixels;
  raw_subpixels.reserve(rows);

  for (int y = 0; y < rows; ++y) {
    const double* row_ptr = phase_f64.ptr<double>(y);
    const uchar* mask_ptr = has_mask ? mask.ptr<uchar>(y) : nullptr;

    double best_x = -1.0;
    double best_diff_metric = 1e9;

    for (int x = 0; x < cols - 1; ++x) {
      if (has_mask && (!mask_ptr[x] || !mask_ptr[x + 1])) continue;

      const double p0 = row_ptr[x];
      const double p1 = row_ptr[x + 1];

      if (!std::isfinite(p0) || !std::isfinite(p1)) continue;

      if (p0 <= target_psi && target_psi < p1) {
        const double diff = p1 - p0;
        if (diff >= min_phys_grad && diff <= max_phys_grad) {
          const double metric = std::abs(diff - 0.065);
          if (metric < best_diff_metric) {
            best_diff_metric = metric;
            best_x = static_cast<double>(x) + (target_psi - p0) / diff;
          }
        }
      }
    }

    if (best_x >= 0.0) {
      raw_subpixels.emplace_back(best_x, static_cast<double>(y));
    }
  }

  if (raw_subpixels.size() < 10) return raw_subpixels;

  std::vector<cv::Point2d> clean_subpixels;
  clean_subpixels.reserve(raw_subpixels.size());

  const int window = 5;
  const double max_lateral_jump = 5.0;

  for (std::size_t i = 0; i < raw_subpixels.size(); ++i) {
    std::vector<double> local_xs;
    for (int w = -window; w <= window; ++w) {
      const int idx = static_cast<int>(i) + w;
      if (idx >= 0 && idx < static_cast<int>(raw_subpixels.size())) {
        local_xs.push_back(raw_subpixels[idx].x);
      }
    }

    std::nth_element(local_xs.begin(), local_xs.begin() + local_xs.size() / 2,
                     local_xs.end());
    const double median_x = local_xs[local_xs.size() / 2];

    if (std::abs(raw_subpixels[i].x - median_x) <= max_lateral_jump) {
      clean_subpixels.push_back(raw_subpixels[i]);
    }
  }

  return clean_subpixels;
}

std::vector<cv::Vec3d> projectSubpixelsToBoard(
    const std::vector<cv::Point2d>& subpixels, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const BoardPlane& board_plane,
    double min_depth_mm, double max_depth_mm) {
  if (subpixels.empty()) return {};

  std::vector<cv::Point2d> norm_points;
  cv::undistortPoints(subpixels, norm_points, camera_matrix, dist_coeffs);

  std::vector<cv::Vec3d> X_ref;
  X_ref.reserve(subpixels.size());

  for (const auto& pt : norm_points) {
    cv::Vec3d ray(pt.x, pt.y, 1.0);
    ray = cv::normalize(ray);

    const double denom = board_plane.normal.dot(ray);
    if (std::abs(denom) < 1e-4) continue;

    const double depth = -board_plane.d / denom;
    if (depth < min_depth_mm || depth > max_depth_mm) continue;

    X_ref.push_back(ray * depth);
  }

  return X_ref;
}

DiscretePlane fitPlaneRobustTLS(const std::vector<cv::Vec3d>& points,
                                double psi,
                                const MsmCalibrationOptions& options) {
  DiscretePlane plane;
  plane.psi = psi;
  plane.point_count = static_cast<int>(points.size());
  plane.valid = false;

  if (points.size() < 30) return plane;

  cv::Vec3d center(0.0, 0.0, 0.0);
  for (const auto& pt : points) center += pt;
  center /= static_cast<double>(points.size());

  cv::Mat A(static_cast<int>(points.size()), 3, CV_64F);
  for (std::size_t i = 0; i < points.size(); ++i) {
    A.at<double>(static_cast<int>(i), 0) = points[i][0] - center[0];
    A.at<double>(static_cast<int>(i), 1) = points[i][1] - center[1];
    A.at<double>(static_cast<int>(i), 2) = points[i][2] - center[2];
  }

  cv::Mat w, u, vt;
  cv::SVD::compute(A, w, u, vt);

  const double s1 = w.at<double>(0);
  const double s2 = w.at<double>(1);
  if (s2 / (s1 + 1e-9) < options.min_spread_ratio) return plane;

  cv::Vec3d normal(vt.at<double>(2, 0), vt.at<double>(2, 1),
                   vt.at<double>(2, 2));
  normal = cv::normalize(normal);
  double d = -normal.dot(center);

  const int max_iters = 5;
  const double huber_k = 1.345;

  for (int iter = 0; iter < max_iters; ++iter) {
    std::vector<double> residuals(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      residuals[i] = std::abs(normal.dot(points[i]) + d);
    }

    std::vector<double> sorted_res = residuals;
    std::nth_element(sorted_res.begin(),
                     sorted_res.begin() + sorted_res.size() / 2,
                     sorted_res.end());
    const double mad = sorted_res[sorted_res.size() / 2];
    const double sigma = std::max(1.4826 * mad, 0.05);

    double sum_weights = 0.0;
    cv::Vec3d w_center(0.0, 0.0, 0.0);
    std::vector<double> weights(points.size(), 1.0);

    for (std::size_t i = 0; i < points.size(); ++i) {
      const double r = residuals[i] / sigma;
      double w_val = (r > huber_k) ? (huber_k / r) : 1.0;
      weights[i] = w_val;
      w_center += points[i] * w_val;
      sum_weights += w_val;
    }

    if (sum_weights < 1e-6) break;
    w_center /= sum_weights;

    cv::Mat W_A(static_cast<int>(points.size()), 3, CV_64F);
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double sqrt_w = std::sqrt(weights[i]);
      W_A.at<double>(static_cast<int>(i), 0) =
          sqrt_w * (points[i][0] - w_center[0]);
      W_A.at<double>(static_cast<int>(i), 1) =
          sqrt_w * (points[i][1] - w_center[1]);
      W_A.at<double>(static_cast<int>(i), 2) =
          sqrt_w * (points[i][2] - w_center[2]);
    }

    cv::Mat w_mat, u_mat, vt_mat;
    cv::SVD::compute(W_A, w_mat, u_mat, vt_mat);

    normal = cv::Vec3d(vt_mat.at<double>(2, 0), vt_mat.at<double>(2, 1),
                       vt_mat.at<double>(2, 2));
    normal = cv::normalize(normal);
    d = -normal.dot(w_center);
  }

  double sum_sq = 0.0;
  int inlier_count = 0;
  for (const auto& pt : points) {
    const double dist = normal.dot(pt) + d;
    if (std::abs(dist) < 0.8) {
      sum_sq += dist * dist;
      inlier_count++;
    }
  }

  const double rms = (inlier_count > 0)
                         ? std::sqrt(sum_sq / static_cast<double>(inlier_count))
                         : 10.0;

  plane.normal = normal;
  plane.d = d;
  plane.rms_mm = rms;
  plane.valid = (rms <= options.max_plane_rms_mm);

  // WLS 综合权重
  plane.weight = static_cast<double>(inlier_count) / (rms * rms + 1e-4);

  return plane;
}

void enforceNormalConsistency(std::vector<DiscretePlane>& planes) {
  if (planes.empty()) return;
  cv::Vec3d ref_normal;
  bool found = false;
  for (const auto& p : planes) {
    if (p.valid) {
      ref_normal = p.normal;
      found = true;
      break;
    }
  }
  if (!found) return;

  for (auto& p : planes) {
    if (p.valid && p.normal.dot(ref_normal) < 0.0) {
      p.normal = -p.normal;
      p.d = -p.d;
    }
  }
}

bool solveNominalRotationGeometry(const std::vector<DiscretePlane>& planes,
                                  double ref_psi, cv::Vec3d& out_w,
                                  cv::Vec3d& out_S0, cv::Vec3d& out_n0,
                                  cv::Vec3d& out_u, cv::Vec3d& out_v) {
  std::vector<DiscretePlane> valid_planes;
  for (const auto& p : planes) {
    if (p.valid) valid_planes.push_back(p);
  }
  if (valid_planes.size() < 3) return false;

  cv::Mat M_cov = cv::Mat::zeros(3, 3, CV_64F);
  for (const auto& vp : valid_planes) {
    const cv::Mat n_i =
        (cv::Mat_<double>(3, 1) << vp.normal[0], vp.normal[1], vp.normal[2]);
    M_cov += vp.weight * (n_i * n_i.t());
  }

  cv::Mat eig_vals, eig_vecs;
  cv::eigen(M_cov, eig_vals, eig_vecs);

  out_w = cv::Vec3d(eig_vecs.at<double>(2, 0), eig_vecs.at<double>(2, 1),
                    eig_vecs.at<double>(2, 2));
  out_w = cv::normalize(out_w);
  if (out_w[1] < 0.0) out_w = -out_w;

  cv::Vec3d a =
      (std::abs(out_w[0]) > 0.9) ? cv::Vec3d(0, 1, 0) : cv::Vec3d(1, 0, 0);
  out_u = cv::normalize(out_w.cross(a));
  out_v = cv::normalize(out_w.cross(out_u));

  cv::Mat A_s0(static_cast<int>(valid_planes.size()), 2, CV_64F);
  cv::Mat b_s0(static_cast<int>(valid_planes.size()), 1, CV_64F);

  for (std::size_t i = 0; i < valid_planes.size(); ++i) {
    const double sqrt_w = std::sqrt(valid_planes[i].weight);
    const auto& n = valid_planes[i].normal;
    A_s0.at<double>(static_cast<int>(i), 0) = sqrt_w * n.dot(out_u);
    A_s0.at<double>(static_cast<int>(i), 1) = sqrt_w * n.dot(out_v);
    b_s0.at<double>(static_cast<int>(i), 0) = -sqrt_w * valid_planes[i].d;
  }

  cv::Mat alpha_beta;
  if (!cv::solve(A_s0, b_s0, alpha_beta, cv::DECOMP_SVD)) return false;

  out_S0 = alpha_beta.at<double>(0) * out_u + alpha_beta.at<double>(1) * out_v;

  double min_diff = 1e9;
  for (const auto& p : valid_planes) {
    const double diff = std::abs(p.psi - ref_psi);
    if (diff < min_diff) {
      min_diff = diff;
      out_n0 = p.normal;
    }
  }
  out_n0 = cv::normalize(out_n0 - out_n0.dot(out_w) * out_w);
  return true;
}

std::vector<double> computeRelativeAngles(
    const std::vector<DiscretePlane>& planes, const cv::Vec3d& w,
    const cv::Vec3d& n0, bool enforce_monotonic) {
  std::vector<double> thetas(planes.size(), 0.0);

  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (!planes[i].valid) continue;

    const cv::Vec3d n_proj =
        cv::normalize(planes[i].normal - planes[i].normal.dot(w) * w);
    const double cos_t = std::clamp(n0.dot(n_proj), -1.0, 1.0);
    const cv::Vec3d cross_n = n0.cross(n_proj);
    thetas[i] = std::atan2(cross_n.dot(w), cos_t);
  }

  if (enforce_monotonic) {
    for (std::size_t i = 1; i < thetas.size(); ++i) {
      if (planes[i].valid && planes[i - 1].valid && thetas[i] < thetas[i - 1]) {
        thetas[i] = thetas[i - 1];
      }
    }
  }

  return thetas;
}

RationalAngleModel fitRationalAngleModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    double psi_ref) {
  RationalAngleModel model;
  model.psi_ref = psi_ref;
  model.valid = false;

  std::vector<int> valid_indices;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (planes[i].valid) valid_indices.push_back(static_cast<int>(i));
  }
  if (valid_indices.size() < 3) return model;

  cv::Mat A(static_cast<int>(valid_indices.size()), 2, CV_64F);
  cv::Mat b(static_cast<int>(valid_indices.size()), 1, CV_64F);

  for (std::size_t i = 0; i < valid_indices.size(); ++i) {
    const int idx = valid_indices[i];
    const double psi = planes[idx].psi;
    const double tan_th = std::tan(thetas[idx]);

    const double cos_th = std::cos(thetas[idx]);
    const double weight = cos_th * cos_th;

    A.at<double>(static_cast<int>(i), 0) = weight * tan_th * psi;
    A.at<double>(static_cast<int>(i), 1) = weight * tan_th;
    b.at<double>(static_cast<int>(i), 0) = weight * (psi - psi_ref);
  }

  cv::Mat params;
  if (cv::solve(A, b, params, cv::DECOMP_SVD)) {
    model.a1 = params.at<double>(0);
    model.a2 = params.at<double>(1);
    model.valid = true;
  }

  return model;
}

// 包含常数项的线性加权最小二乘求解：order=2 时解 10 个参数
HarmonicDriftModel fitHarmonicDriftModel(
    const std::vector<DiscretePlane>& planes, const std::vector<double>& thetas,
    const cv::Vec3d& w, const cv::Vec3d& S0, const cv::Vec3d& u,
    const cv::Vec3d& v, int order) {
  HarmonicDriftModel model;
  model.order = order;
  model.valid = false;

  if (order <= 0) return model;

  std::vector<int> valid_indices;
  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (planes[i].valid) valid_indices.push_back(static_cast<int>(i));
  }

  // 未知数数量：常数项 (2个) + 谐波项 (4 * order 个)
  const int num_vars = 2 + 4 * order;
  if (static_cast<int>(valid_indices.size()) < num_vars + 2) return model;

  const int num_pts = static_cast<int>(valid_indices.size());
  cv::Mat H = cv::Mat::zeros(num_pts, num_vars, CV_64F);
  cv::Mat b_res(num_pts, 1, CV_64F);

  for (int i = 0; i < num_pts; ++i) {
    const int idx = valid_indices[i];
    const double th = thetas[idx];
    const auto& n = planes[idx].normal;
    const double d = planes[idx].d;

    const double nu = n.dot(u);
    const double nv = n.dot(v);

    // 列 0 与 1：常数项基底 (分别对应 beta_u[0] 和 beta_v[0])
    H.at<double>(i, 0) = nu;
    H.at<double>(i, 1) = nv;

    // 后续列：各阶谐波项基底
    for (int k = 1; k <= order; ++k) {
      const double ang = static_cast<double>(k) * th;
      const double c = std::cos(ang);
      const double s = std::sin(ang);

      const int col_base = 2 + 4 * (k - 1);
      H.at<double>(i, col_base + 0) = nu * c;
      H.at<double>(i, col_base + 1) = nu * s;
      H.at<double>(i, col_base + 2) = nv * c;
      H.at<double>(i, col_base + 3) = nv * s;
    }

    b_res.at<double>(i, 0) = -(n.dot(S0) + d);
  }

  // 引入微弱 Tikhonov 岭正则项稳定窄视场条件数
  cv::Mat HtH = H.t() * H;
  const double lambda = 1e-3;
  for (int j = 0; j < num_vars; ++j) {
    HtH.at<double>(j, j) += lambda;
  }
  cv::Mat Htb = H.t() * b_res;

  cv::Mat beta;
  if (cv::solve(HtH, Htb, beta, cv::DECOMP_CHOLESKY)) {
    // 每个方向包含 1 个常数项 + 2*order 个谐波项，总长 2*order + 1
    model.beta_u.resize(2 * order + 1, 0.0);
    model.beta_v.resize(2 * order + 1, 0.0);

    // 0 号索引存放常数偏置
    model.beta_u[0] = beta.at<double>(0);
    model.beta_v[0] = beta.at<double>(1);

    // 交替存入各频阶的 cos 与 sin 系数
    for (int k = 1; k <= order; ++k) {
      const int col_base = 2 + 4 * (k - 1);
      const int dst_base = 1 + 2 * (k - 1);

      model.beta_u[dst_base + 0] = beta.at<double>(col_base + 0);
      model.beta_u[dst_base + 1] = beta.at<double>(col_base + 1);
      model.beta_v[dst_base + 0] = beta.at<double>(col_base + 2);
      model.beta_v[dst_base + 1] = beta.at<double>(col_base + 3);
    }
    model.valid = true;
  }

  return model;
}

double evaluateMsmReconstruction(const MsmCalibrationResult& result,
                                 const cv::Mat& phase_map,
                                 const cv::Mat& camera_matrix,
                                 const cv::Mat& dist_coeffs,
                                 const BoardPlane& board_plane, double min_psi,
                                 double max_psi, int pixel_stride) {
  if (phase_map.empty()) return 0.0;

  cv::Mat p64;
  if (phase_map.channels() > 1) {
    cv::extractChannel(phase_map, p64, 0);
  } else {
    p64 = phase_map;
  }
  p64.convertTo(p64, CV_64F);

  std::vector<cv::Point2d> subpixels;
  std::vector<double> psis;

  for (int y = 0; y < p64.rows; y += pixel_stride) {
    const double* r_ptr = p64.ptr<double>(y);
    for (int x = 0; x < p64.cols; x += pixel_stride) {
      const double val = r_ptr[x];
      if (std::isfinite(val) && val >= min_psi && val <= max_psi) {
        subpixels.emplace_back(static_cast<double>(x), static_cast<double>(y));
        psis.push_back(val);
      }
    }
  }

  if (subpixels.empty()) return 0.0;

  std::vector<cv::Point2d> norm_pts;
  cv::undistortPoints(subpixels, norm_pts, camera_matrix, dist_coeffs);

  double sum_sq_err = 0.0;
  int count = 0;

  for (std::size_t i = 0; i < norm_pts.size(); ++i) {
    cv::Vec3d ray(norm_pts[i].x, norm_pts[i].y, 1.0);
    ray = cv::normalize(ray);

    const double denom_board = board_plane.normal.dot(ray);
    if (std::abs(denom_board) < 1e-4) continue;
    const double depth_gt = -board_plane.d / denom_board;
    if (depth_gt < 120.0 || depth_gt > 250.0) continue;
    const cv::Vec3d X_gt = ray * depth_gt;

    cv::Vec3d n_pred;
    double d_pred = 0.0;
    result.evaluatePlane(psis[i], n_pred, d_pred);

    const double denom_pred = n_pred.dot(ray);
    if (std::abs(denom_pred) < 1e-4) continue;
    const double depth_pred = -d_pred / denom_pred;
    const cv::Vec3d X_pred = ray * depth_pred;

    const double err = cv::norm(X_pred - X_gt);
    sum_sq_err += err * err;
    count++;
  }

  return (count > 0) ? std::sqrt(sum_sq_err / static_cast<double>(count)) : 0.0;
}

MsmCalibrationResult calibrateMsm(const MsmCalibrationConfig& config,
                                  const CameraCalibrationResult& camera_calib,
                                  const std::vector<cv::Mat>& all_phase_maps) {
  std::cout << "\nStarting MSM calibration pipeline..." << std::endl;

  const std::size_t train_count = config.train_poses.size();
  std::vector<BoardPlane> train_board_planes;
  std::vector<cv::Mat> train_phase_maps;

  train_board_planes.reserve(train_count);
  train_phase_maps.reserve(train_count);

  std::cout << "[Step 1] Preparing board planes for " << train_count
            << " training poses..." << std::endl;

  for (std::size_t i = 0; i < train_count; ++i) {
    const int pid = config.train_poses[i];
    if (pid < 0 ||
        pid >= static_cast<int>(camera_calib.rotation_vectors.size()) ||
        pid >= static_cast<int>(all_phase_maps.size())) {
      throw std::runtime_error("Pose ID out of bounds: " + std::to_string(pid));
    }

    auto bp = computeBoardPlane(camera_calib.rotation_vectors[pid],
                                camera_calib.translation_vectors[pid]);
    train_board_planes.push_back(bp);
    train_phase_maps.push_back(all_phase_maps[pid]);

    std::cout << "  Pose " << std::setw(2) << pid << ": board normal=["
              << bp.normal[0] << ", " << bp.normal[1] << ", " << bp.normal[2]
              << "], distance d=" << bp.d << " mm" << std::endl;
  }

  double eff_min_psi = config.options.min_psi;
  double eff_max_psi = config.options.max_psi;

  if (config.options.auto_phase_range) {
    std::cout << "[Step 2] Auto-detecting multi-view co-visible phase range ("
              << "min_covisible_poses=" << config.options.min_covisible_poses
              << ") ..." << std::endl;
    const auto auto_range = autoDetectValidPhaseRange(
        train_phase_maps, config.options.min_covisible_poses);
    eff_min_psi = auto_range.first;
    eff_max_psi = auto_range.second;
    std::cout << "  Auto-detected co-visible range: [" << eff_min_psi << ", "
              << eff_max_psi << "] rad (span: " << eff_max_psi - eff_min_psi
              << " rad)" << std::endl;
  } else {
    std::cout << "[Step 2] Using configured phase range: [" << eff_min_psi
              << ", " << eff_max_psi << "] rad" << std::endl;
  }

  const int num_planes = std::max(10, config.options.plane_count);
  std::vector<DiscretePlane> discrete_planes;
  discrete_planes.reserve(num_planes);

  std::cout << "[Step 3] Extracting 1D subpixels and fitting " << num_planes
            << " discrete planes..." << std::endl;

  for (int p_idx = 0; p_idx < num_planes; ++p_idx) {
    const double target_psi =
        eff_min_psi + static_cast<double>(p_idx) * (eff_max_psi - eff_min_psi) /
                          static_cast<double>(num_planes - 1);

    std::vector<cv::Vec3d> pooled_pts;

    for (std::size_t i = 0; i < train_count; ++i) {
      const auto subpixels =
          extractIsoPhaseSubpixels(train_phase_maps[i], target_psi);
      if (subpixels.empty()) continue;

      const auto X_ref =
          projectSubpixelsToBoard(subpixels, camera_calib.camera_matrix,
                                  camera_calib.distortion_coefficients,
                                  train_board_planes[i], 120.0, 250.0);
      pooled_pts.insert(pooled_pts.end(), X_ref.begin(), X_ref.end());
    }

    auto plane = fitPlaneRobustTLS(pooled_pts, target_psi, config.options);
    discrete_planes.push_back(plane);

    if (p_idx == 0 || p_idx == num_planes / 2 || p_idx == num_planes - 1) {
      std::cout << "  Plane " << std::setw(2) << p_idx << " (psi=" << std::fixed
                << std::setprecision(2) << target_psi
                << "): pts=" << plane.point_count
                << ", rms=" << std::setprecision(5) << plane.rms_mm
                << " mm, valid=" << (plane.valid ? "YES" : "NO") << std::endl;
    }
  }

  int valid_count = 0;
  for (const auto& p : discrete_planes) {
    if (p.valid) valid_count++;
  }
  std::cout << "Total valid planes extracted: " << valid_count << " / "
            << num_planes << std::endl;

  if (valid_count < 3) {
    throw std::runtime_error("Insufficient valid planes for MSM calibration.");
  }

  enforceNormalConsistency(discrete_planes);

  const double ref_psi = (eff_min_psi + eff_max_psi) * 0.5;
  MsmCalibrationResult result;

  if (!solveNominalRotationGeometry(
          discrete_planes, ref_psi, result.nominal_axis_w,
          result.nominal_center_s0, result.ref_normal_n0, result.basis_u,
          result.basis_v)) {
    throw std::runtime_error("Failed to solve nominal rotation geometry.");
  }

  const auto thetas = computeRelativeAngles(
      discrete_planes, result.nominal_axis_w, result.ref_normal_n0, true);
  result.angle_model = fitRationalAngleModel(discrete_planes, thetas, ref_psi);

  result.harmonic_drift = fitHarmonicDriftModel(
      discrete_planes, thetas, result.nominal_axis_w, result.nominal_center_s0,
      result.basis_u, result.basis_v, config.options.harmonic_order);

  // 训练集闭环 3D 误差评估
  double sum_train_3d_err = 0.0;
  int train_eval_count = 0;
  for (std::size_t i = 0; i < train_count; ++i) {
    const double pose_err = evaluateMsmReconstruction(
        result, train_phase_maps[i], camera_calib.camera_matrix,
        camera_calib.distortion_coefficients, train_board_planes[i],
        eff_min_psi, eff_max_psi, 8);
    if (pose_err > 0.0) {
      sum_train_3d_err += pose_err * pose_err;
      train_eval_count++;
    }
  }
  result.train_rmse_mm =
      (train_eval_count > 0)
          ? std::sqrt(sum_train_3d_err / static_cast<double>(train_eval_count))
          : 0.0;

  // 测试集闭环 3D 误差评估
  if (!config.test_poses.empty()) {
    const int test_pid = config.test_poses[0];
    if (test_pid >= 0 && test_pid < static_cast<int>(all_phase_maps.size())) {
      const auto test_bp =
          computeBoardPlane(camera_calib.rotation_vectors[test_pid],
                            camera_calib.translation_vectors[test_pid]);
      result.test_rmse_mm = evaluateMsmReconstruction(
          result, all_phase_maps[test_pid], camera_calib.camera_matrix,
          camera_calib.distortion_coefficients, test_bp, eff_min_psi,
          eff_max_psi, 4);
    }
  }

  std::cout << "\n================ Calibration Summary ================"
            << std::endl;
  std::cout << "Nominal Axis (w): [" << result.nominal_axis_w[0] << ", "
            << result.nominal_axis_w[1] << ", " << result.nominal_axis_w[2]
            << "]" << std::endl;
  std::cout << "Nominal Center (S0): [" << result.nominal_center_s0[0] << ", "
            << result.nominal_center_s0[1] << ", "
            << result.nominal_center_s0[2] << "] mm" << std::endl;
  std::cout << "Rational Angle Model: tan(theta) = (psi - " << ref_psi
            << ") / (" << result.angle_model.a1 << "*psi + "
            << result.angle_model.a2 << ")" << std::endl;
  std::cout << "Harmonic Drift Order: " << result.harmonic_drift.order
            << " (valid=" << (result.harmonic_drift.valid ? "YES" : "NO") << ")"
            << std::endl;
  if (result.harmonic_drift.valid && result.harmonic_drift.order > 0) {
    std::cout << "Harmonic DC Offset: c_u = " << result.harmonic_drift.beta_u[0]
              << " mm, c_v = " << result.harmonic_drift.beta_v[0] << " mm"
              << std::endl;
  }
  std::cout << "Closed-Loop TRAIN 3D RMSE: " << result.train_rmse_mm << " mm"
            << std::endl;
  std::cout << "Closed-Loop TEST 3D RMSE:  " << result.test_rmse_mm << " mm"
            << std::endl;
  std::cout << "=====================================================\n"
            << std::endl;

  return result;
}

bool saveMsmCalibrationResult(const std::string& file_path,
                              const MsmCalibrationResult& result) {
  cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) return false;

  fs << "nominal_axis_w" << cv::Mat(result.nominal_axis_w);
  fs << "nominal_center_s0" << cv::Mat(result.nominal_center_s0);
  fs << "ref_normal_n0" << cv::Mat(result.ref_normal_n0);
  fs << "basis_u" << cv::Mat(result.basis_u);
  fs << "basis_v" << cv::Mat(result.basis_v);

  fs << "psi_ref" << result.angle_model.psi_ref;
  fs << "angle_model_a1" << result.angle_model.a1;
  fs << "angle_model_a2" << result.angle_model.a2;

  fs << "harmonic_order" << result.harmonic_drift.order;
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

  MsmCalibrationResult res;
  cv::Mat w_mat, s0_mat, n0_mat, u_mat, v_mat;

  fs["nominal_axis_w"] >> w_mat;
  fs["nominal_center_s0"] >> s0_mat;
  fs["ref_normal_n0"] >> n0_mat;
  fs["basis_u"] >> u_mat;
  fs["basis_v"] >> v_mat;

  res.nominal_axis_w = cv::Vec3d(w_mat);
  res.nominal_center_s0 = cv::Vec3d(s0_mat);
  res.ref_normal_n0 = cv::Vec3d(n0_mat);
  res.basis_u = cv::Vec3d(u_mat);
  res.basis_v = cv::Vec3d(v_mat);

  fs["psi_ref"] >> res.angle_model.psi_ref;
  fs["angle_model_a1"] >> res.angle_model.a1;
  fs["angle_model_a2"] >> res.angle_model.a2;
  res.angle_model.valid = true;

  fs["harmonic_order"] >> res.harmonic_drift.order;
  fs["beta_u"] >> res.harmonic_drift.beta_u;
  fs["beta_v"] >> res.harmonic_drift.beta_v;
  res.harmonic_drift.valid = true;

  fs["train_rmse_mm"] >> res.train_rmse_mm;
  fs["test_rmse_mm"] >> res.test_rmse_mm;

  fs.release();
  return res;
}

}  // namespace msm3d
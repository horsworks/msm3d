#include "msm3d/msm/iso_phase.hpp"

#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace msm3d {
namespace {

struct LocalPhaseFit {
  bool valid = false;
  double x = 0.0;
  double slope = 0.0;
  double rmse = std::numeric_limits<double>::infinity();
};

LocalPhaseFit fitLocalPhaseLine(const cv::Mat& phase_f64, int y, int crossing_x,
                                double target_psi, const cv::Mat& mask,
                                int half_window, double min_abs_gradient,
                                double max_abs_gradient) {
  LocalPhaseFit result;

  const int cols = phase_f64.cols;
  const bool has_mask = !mask.empty() && mask.size() == phase_f64.size();

  // Include both crossing endpoints and a symmetric neighborhood.
  const int x_begin = std::max(0, crossing_x - half_window);
  const int x_end = std::min(cols - 1, crossing_x + 1 + half_window);

  const double* row_ptr = phase_f64.ptr<double>(y);
  const uchar* mask_ptr = has_mask ? mask.ptr<uchar>(y) : nullptr;

  double sum_x = 0.0;
  double sum_p = 0.0;
  int count = 0;

  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    sum_x += static_cast<double>(x);
    sum_p += p;
    ++count;
  }

  if (count < 3) {
    return result;
  }

  const double mean_x = sum_x / static_cast<double>(count);
  const double mean_p = sum_p / static_cast<double>(count);

  double sxx = 0.0;
  double sxp = 0.0;

  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    const double dx = static_cast<double>(x) - mean_x;
    const double dp = p - mean_p;
    sxx += dx * dx;
    sxp += dx * dp;
  }

  if (sxx <= kEpsilon) {
    return result;
  }

  const double slope = sxp / sxx;
  const double abs_slope = std::abs(slope);
  if (!std::isfinite(slope) || abs_slope < min_abs_gradient ||
      abs_slope > max_abs_gradient) {
    return result;
  }

  const double intercept = mean_p - slope * mean_x;
  const double fitted_x = (target_psi - intercept) / slope;

  // The fitted crossing must remain near the local support interval.
  if (!std::isfinite(fitted_x) ||
      fitted_x < static_cast<double>(x_begin) - 0.5 ||
      fitted_x > static_cast<double>(x_end) + 0.5) {
    return result;
  }

  double sum_sq = 0.0;
  int residual_count = 0;
  for (int x = x_begin; x <= x_end; ++x) {
    if (has_mask && !mask_ptr[x]) {
      continue;
    }

    const double p = row_ptr[x];
    if (!std::isfinite(p)) {
      continue;
    }

    const double predicted = slope * static_cast<double>(x) + intercept;
    const double residual = p - predicted;
    sum_sq += residual * residual;
    ++residual_count;
  }

  if (residual_count < 3) {
    return result;
  }

  result.valid = true;
  result.x = fitted_x;
  result.slope = slope;
  result.rmse = std::sqrt(sum_sq / static_cast<double>(residual_count));
  return result;
}

}  // namespace

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

  double global_min = std::numeric_limits<double>::infinity();
  double global_max = -std::numeric_limits<double>::infinity();
  for (const auto& interval : intervals) {
    global_min = std::min(global_min, interval.p_low);
    global_max = std::max(global_max, interval.p_high);
  }

  const int required_poses = std::max(
      2, std::min(min_covisible_poses, static_cast<int>(intervals.size())));

  constexpr double step = 0.5;
  double best_start = 0.0;
  double best_end = 0.0;
  double cur_start = -1.0;
  double max_len = 0.0;

  for (double psi = global_min; psi <= global_max; psi += step) {
    int covisible_count = 0;
    for (const auto& interval : intervals) {
      if (psi >= interval.p_low && psi <= interval.p_high) {
        ++covisible_count;
      }
    }

    if (covisible_count >= required_poses) {
      if (cur_start < 0.0) cur_start = psi;
    } else if (cur_start >= 0.0) {
      const double len = (psi - step) - cur_start;
      if (len > max_len) {
        max_len = len;
        best_start = cur_start;
        best_end = psi - step;
      }
      cur_start = -1.0;
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
  if (best_end <= best_start) {
    return {global_min, global_max};
  }

  return {best_start, best_end};
}

std::vector<cv::Point2d> extractIsoPhaseSubpixels(
    const cv::Mat& phase_map, double target_psi, const cv::Mat& mask,
    double grad_threshold, int fit_half_window, double max_grad_threshold,
    double max_lateral_jump_px) {
  if (phase_map.empty()) return {};

  if (fit_half_window < 1) {
    throw std::invalid_argument("fit_half_window must be at least 1.");
  }

  const double min_abs_gradient = std::max(grad_threshold, 1e-9);
  if (!(max_grad_threshold > min_abs_gradient)) {
    throw std::invalid_argument(
        "max_grad_threshold must be greater than grad_threshold.");
  }

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
  const bool has_mask = !mask.empty() && mask.size() == phase_f64.size();

  std::vector<cv::Point2d> raw_subpixels;
  raw_subpixels.reserve(rows);

  for (int y = 0; y < rows; ++y) {
    const double* row_ptr = phase_f64.ptr<double>(y);
    const uchar* mask_ptr = has_mask ? mask.ptr<uchar>(y) : nullptr;

    LocalPhaseFit best_fit;

    for (int x = 0; x < cols - 1; ++x) {
      if (has_mask && (!mask_ptr[x] || !mask_ptr[x + 1])) {
        continue;
      }

      const double p0 = row_ptr[x];
      const double p1 = row_ptr[x + 1];
      if (!std::isfinite(p0) || !std::isfinite(p1)) {
        continue;
      }

      const bool crossing = (p0 <= target_psi && target_psi <= p1) ||
                            (p1 <= target_psi && target_psi <= p0);
      if (!crossing || std::abs(p1 - p0) < min_abs_gradient) {
        continue;
      }

      const LocalPhaseFit candidate =
          fitLocalPhaseLine(phase_f64, y, x, target_psi, mask, fit_half_window,
                            min_abs_gradient, max_grad_threshold);

      if (!candidate.valid) {
        continue;
      }

      if (!best_fit.valid || candidate.rmse < best_fit.rmse) {
        best_fit = candidate;
      }
    }

    if (best_fit.valid) {
      raw_subpixels.emplace_back(best_fit.x, static_cast<double>(y));
    }
  }

  if (raw_subpixels.size() < 10 || max_lateral_jump_px <= 0.0) {
    return raw_subpixels;
  }

  // Suppress isolated row-wise x outliers while preserving the smooth
  // iso-phase curve.
  std::vector<cv::Point2d> clean_subpixels;
  clean_subpixels.reserve(raw_subpixels.size());

  constexpr int kRowMedianHalfWindow = 5;
  for (std::size_t i = 0; i < raw_subpixels.size(); ++i) {
    std::vector<double> local_xs;
    local_xs.reserve(2 * kRowMedianHalfWindow + 1);

    for (int w = -kRowMedianHalfWindow; w <= kRowMedianHalfWindow; ++w) {
      const int idx = static_cast<int>(i) + w;
      if (idx >= 0 && idx < static_cast<int>(raw_subpixels.size())) {
        local_xs.push_back(raw_subpixels[static_cast<std::size_t>(idx)].x);
      }
    }

    const std::size_t mid = local_xs.size() / 2;
    std::nth_element(local_xs.begin(),
                     local_xs.begin() + static_cast<std::ptrdiff_t>(mid),
                     local_xs.end());
    const double median_x = local_xs[mid];

    if (std::abs(raw_subpixels[i].x - median_x) <= max_lateral_jump_px) {
      clean_subpixels.push_back(raw_subpixels[i]);
    }
  }

  return clean_subpixels;
}

}  // namespace msm3d

#include "msm3d/phase_processor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace msm3d {
namespace {

constexpr double kTwoPi = 2.0 * CV_PI;

void convertToGray64(const cv::Mat& image, cv::Mat& gray64,
                     double& sensor_max_value) {
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  switch (gray.depth()) {
    case CV_8U:
      sensor_max_value = 255.0;
      break;
    case CV_16U:
      sensor_max_value = 65535.0;
      break;
    default:
      sensor_max_value = 0.0;
      break;
  }

  gray.convertTo(gray64, CV_64F);
}

double circularPhaseError(double a, double b) {
  const double delta = a - b;
  return std::abs(std::atan2(std::sin(delta), std::cos(delta)));
}

bool validateOptionalMaps(const std::vector<cv::Mat>& maps,
                          const cv::Size& size) {
  if (maps.size() != 3) {
    return false;
  }

  for (const auto& map : maps) {
    if (map.empty() || map.size() != size || map.type() != CV_64F) {
      return false;
    }
  }

  return true;
}

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }

  q = std::clamp(q, 0.0, 1.0);
  const double position = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));

  std::nth_element(values.begin(), values.begin() + lower, values.end());
  const double lower_value = values[lower];

  if (upper == lower) {
    return lower_value;
  }

  std::nth_element(values.begin(), values.begin() + upper, values.end());
  const double upper_value = values[upper];
  const double t = position - static_cast<double>(lower);
  return lower_value + t * (upper_value - lower_value);
}

double phaseFusionWeight(double fit_residual_ratio, double saturation_fraction,
                         const PhaseQualityOptions& options) {
  const double residual =
      std::max(std::abs(fit_residual_ratio), options.fusion_fit_residual_floor);

  const double saturation_score = std::clamp(
      1.0 - saturation_fraction / options.max_saturation_fraction, 0.0, 1.0);

  return saturation_score * saturation_score / (residual * residual);
}

double fuseUnwrappedPhases(double abs1, double abs2, double abs3, int f1,
                           int f2, int f3, double fit1, double fit2,
                           double fit3, double saturation1, double saturation2,
                           double saturation3,
                           const PhaseQualityOptions& options,
                           double& out_disagreement_rad) {
  const double a1 = 1.0;
  const double a2 = static_cast<double>(f2) / static_cast<double>(f1);
  const double a3 = static_cast<double>(f3) / static_cast<double>(f1);

  const double w1 = phaseFusionWeight(fit1, saturation1, options);
  const double w2 = phaseFusionWeight(fit2, saturation2, options);
  const double w3 = phaseFusionWeight(fit3, saturation3, options);

  const double denominator = w1 * a1 * a1 + w2 * a2 * a2 + w3 * a3 * a3;

  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    out_disagreement_rad = 0.0;
    return abs1;
  }

  const double fused =
      (w1 * a1 * abs1 + w2 * a2 * abs2 + w3 * a3 * abs3) / denominator;

  const double candidate1 = abs1;
  const double candidate2 = abs2 / a2;
  const double candidate3 = abs3 / a3;

  const double ew1 = w1 * a1 * a1;
  const double ew2 = w2 * a2 * a2;
  const double ew3 = w3 * a3 * a3;
  const double effective_weight_sum = ew1 + ew2 + ew3;

  const double disagreement_sq =
      (ew1 * (candidate1 - fused) * (candidate1 - fused) +
       ew2 * (candidate2 - fused) * (candidate2 - fused) +
       ew3 * (candidate3 - fused) * (candidate3 - fused)) /
      std::max(effective_weight_sum, 1e-12);

  out_disagreement_rad = std::sqrt(std::max(0.0, disagreement_sq));
  return fused;
}

bool solveLinear3x3(double matrix[3][3], double rhs[3], double solution[3]) {
  double augmented[3][4] = {{matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]},
                            {matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]},
                            {matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]}};

  for (int column = 0; column < 3; ++column) {
    int pivot = column;
    double pivot_abs = std::abs(augmented[column][column]);

    for (int row = column + 1; row < 3; ++row) {
      const double value = std::abs(augmented[row][column]);
      if (value > pivot_abs) {
        pivot = row;
        pivot_abs = value;
      }
    }

    if (!(pivot_abs > 1e-12) || !std::isfinite(pivot_abs)) {
      return false;
    }

    if (pivot != column) {
      for (int k = column; k < 4; ++k) {
        std::swap(augmented[column][k], augmented[pivot][k]);
      }
    }

    const double divisor = augmented[column][column];
    for (int k = column; k < 4; ++k) {
      augmented[column][k] /= divisor;
    }

    for (int row = 0; row < 3; ++row) {
      if (row == column) {
        continue;
      }

      const double factor = augmented[row][column];
      for (int k = column; k < 4; ++k) {
        augmented[row][k] -= factor * augmented[column][k];
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    solution[i] = augmented[i][3];
    if (!std::isfinite(solution[i])) {
      return false;
    }
  }

  return true;
}

struct LocalPlaneFit {
  bool valid = false;
  double ax = 0.0;
  double ay = 0.0;
  double center = 0.0;
  double weighted_rmse = 0.0;
  double mean_confidence = 0.0;
  int valid_count = 0;
};

LocalPlaneFit fitLocalPlane(const cv::Mat& phase, const cv::Mat& confidence,
                            int center_x, int center_y, int radius,
                            double robust_scale_rad, bool robust_pass,
                            const LocalPlaneFit* initial_fit = nullptr) {
  double sw = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  double syy = 0.0;
  double sz = 0.0;
  double sxz = 0.0;
  double syz = 0.0;
  double confidence_sum = 0.0;
  int valid_count = 0;

  for (int dy = -radius; dy <= radius; ++dy) {
    const int y = center_y + dy;
    if (y < 0 || y >= phase.rows) {
      continue;
    }

    const double* phase_row = phase.ptr<double>(y);
    const double* confidence_row = confidence.ptr<double>(y);

    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = center_x + dx;
      if (x < 0 || x >= phase.cols) {
        continue;
      }

      const double z = phase_row[x];
      const double input_confidence = confidence_row[x];

      if (!std::isfinite(z) || !std::isfinite(input_confidence) ||
          input_confidence <= 0.0) {
        continue;
      }

      double weight = std::clamp(input_confidence, 0.0, 1.0);

      if (robust_pass && initial_fit != nullptr && initial_fit->valid &&
          robust_scale_rad > 0.0) {
        const double predicted = initial_fit->ax * static_cast<double>(dx) +
                                 initial_fit->ay * static_cast<double>(dy) +
                                 initial_fit->center;
        const double residual = z - predicted;
        const double abs_residual = std::abs(residual);

        if (abs_residual > robust_scale_rad) {
          weight *= robust_scale_rad / std::max(abs_residual, 1e-12);
        }
      }

      if (!(weight > 0.0)) {
        continue;
      }

      const double dx_d = static_cast<double>(dx);
      const double dy_d = static_cast<double>(dy);

      sw += weight;
      sx += weight * dx_d;
      sy += weight * dy_d;
      sxx += weight * dx_d * dx_d;
      sxy += weight * dx_d * dy_d;
      syy += weight * dy_d * dy_d;
      sz += weight * z;
      sxz += weight * dx_d * z;
      syz += weight * dy_d * z;

      confidence_sum += std::clamp(input_confidence, 0.0, 1.0);
      ++valid_count;
    }
  }

  LocalPlaneFit fit;
  fit.valid_count = valid_count;

  if (valid_count < 3 || !(sw > 0.0)) {
    return fit;
  }

  double matrix[3][3] = {{sxx, sxy, sx}, {sxy, syy, sy}, {sx, sy, sw}};
  double rhs[3] = {sxz, syz, sz};
  double solution[3] = {0.0, 0.0, 0.0};

  if (!solveLinear3x3(matrix, rhs, solution)) {
    return fit;
  }

  fit.ax = solution[0];
  fit.ay = solution[1];
  fit.center = solution[2];
  fit.mean_confidence = confidence_sum / static_cast<double>(valid_count);

  double weighted_residual_sum_sq = 0.0;
  double residual_weight_sum = 0.0;

  for (int dy = -radius; dy <= radius; ++dy) {
    const int y = center_y + dy;
    if (y < 0 || y >= phase.rows) {
      continue;
    }

    const double* phase_row = phase.ptr<double>(y);
    const double* confidence_row = confidence.ptr<double>(y);

    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = center_x + dx;
      if (x < 0 || x >= phase.cols) {
        continue;
      }

      const double z = phase_row[x];
      const double input_confidence = confidence_row[x];

      if (!std::isfinite(z) || !std::isfinite(input_confidence) ||
          input_confidence <= 0.0) {
        continue;
      }

      double weight = std::clamp(input_confidence, 0.0, 1.0);
      const double predicted = fit.ax * static_cast<double>(dx) +
                               fit.ay * static_cast<double>(dy) + fit.center;
      const double residual = z - predicted;

      if (robust_pass && robust_scale_rad > 0.0) {
        const double abs_residual = std::abs(residual);
        if (abs_residual > robust_scale_rad) {
          weight *= robust_scale_rad / std::max(abs_residual, 1e-12);
        }
      }

      weighted_residual_sum_sq += weight * residual * residual;
      residual_weight_sum += weight;
    }
  }

  if (residual_weight_sum > 0.0) {
    fit.weighted_rmse =
        std::sqrt(weighted_residual_sum_sq / residual_weight_sum);
  }

  fit.valid = std::isfinite(fit.center) && std::isfinite(fit.weighted_rmse);
  return fit;
}

}  // namespace

cv::Mat PhaseProcessor::computeWrappedPhase(const std::vector<cv::Mat>& images,
                                            cv::Mat* out_modulation,
                                            cv::Mat* out_fit_residual_ratio,
                                            cv::Mat* out_saturation_fraction) {
  if (images.empty()) {
    throw std::invalid_argument("Input fringe images list is empty.");
  }

  const cv::Size image_size = images[0].size();
  const std::size_t count = images.size();
  const double phase_step = kTwoPi / static_cast<double>(count);

  cv::Mat dc_sum = cv::Mat::zeros(image_size, CV_64F);
  cv::Mat sin_sum = cv::Mat::zeros(image_size, CV_64F);
  cv::Mat cos_sum = cv::Mat::zeros(image_size, CV_64F);
  cv::Mat saturation_count = cv::Mat::zeros(image_size, CV_64F);

  cv::Mat gray64;
  for (std::size_t i = 0; i < count; ++i) {
    if (images[i].size() != image_size) {
      throw std::invalid_argument(
          "All fringe images must share the same resolution.");
    }

    double sensor_max = 0.0;
    convertToGray64(images[i], gray64, sensor_max);

    dc_sum += gray64;

    const double delta = phase_step * static_cast<double>(i);
    cv::scaleAdd(gray64, std::sin(delta), sin_sum, sin_sum);
    cv::scaleAdd(gray64, std::cos(delta), cos_sum, cos_sum);

    if (sensor_max > 0.0 && out_saturation_fraction != nullptr) {
      cv::Mat low_mask;
      cv::Mat high_mask;
      cv::Mat saturated_mask;
      cv::Mat saturated64;

      cv::compare(gray64, 0.01 * sensor_max, low_mask, cv::CMP_LE);
      cv::compare(gray64, 0.99 * sensor_max, high_mask, cv::CMP_GE);
      cv::bitwise_or(low_mask, high_mask, saturated_mask);
      saturated_mask.convertTo(saturated64, CV_64F, 1.0 / 255.0);
      saturation_count += saturated64;
    }
  }

  cv::Mat phase;
  cv::Mat neg_sin_sum = -sin_sum;
  cv::phase(cos_sum, neg_sin_sum, phase, false);

  cv::Mat modulation;
  cv::magnitude(cos_sum, sin_sum, modulation);
  modulation *= 2.0 / static_cast<double>(count);

  if (out_modulation != nullptr) {
    *out_modulation = modulation.clone();
  }

  if (out_fit_residual_ratio != nullptr) {
    const cv::Mat mean = dc_sum / static_cast<double>(count);
    const cv::Mat cos_coefficient =
        cos_sum * (2.0 / static_cast<double>(count));
    const cv::Mat sin_coefficient =
        sin_sum * (2.0 / static_cast<double>(count));

    cv::Mat residual_sum_sq = cv::Mat::zeros(image_size, CV_64F);

    for (std::size_t i = 0; i < count; ++i) {
      double sensor_max = 0.0;
      convertToGray64(images[i], gray64, sensor_max);

      const double delta = phase_step * static_cast<double>(i);
      cv::Mat prediction = mean + cos_coefficient * std::cos(delta) +
                           sin_coefficient * std::sin(delta);
      cv::Mat residual = gray64 - prediction;
      cv::Mat residual_sq;
      cv::multiply(residual, residual, residual_sq);
      residual_sum_sq += residual_sq;
    }

    cv::Mat fit_rmse;
    cv::sqrt(residual_sum_sq / static_cast<double>(count), fit_rmse);

    cv::Mat safe_modulation = modulation + 1e-9;
    cv::divide(fit_rmse, safe_modulation, *out_fit_residual_ratio);
  }

  if (out_saturation_fraction != nullptr) {
    *out_saturation_fraction = saturation_count / static_cast<double>(count);
  }

  return phase;
}

cv::Mat PhaseProcessor::computeAbsolutePhase(
    const std::vector<cv::Mat>& wrapped_phases,
    const std::vector<int>& frequencies) {
  return computeAbsolutePhase(wrapped_phases, frequencies, {}, 0.0);
}

cv::Mat PhaseProcessor::computeAbsolutePhase(
    const std::vector<cv::Mat>& wrapped_phases,
    const std::vector<int>& frequencies,
    const std::vector<cv::Mat>& modulations, double min_modulation) {
  if (wrapped_phases.size() != 3 || frequencies.size() != 3) {
    throw std::invalid_argument(
        "Multi-frequency unwrap expects exactly 3 wrapped phases and 3 "
        "frequencies.");
  }

  const int f1 = frequencies[0];
  const int f2 = frequencies[1];
  const int f3 = frequencies[2];
  const int f12 = f1 - f2;
  const int f23 = f2 - f3;
  const int f123 = f12 - f23;

  if (f12 <= 0 || f23 <= 0 || f123 <= 0) {
    throw std::invalid_argument(
        "Invalid frequency combination: requires f1 > f2 > f3 and "
        "(f1-f2) > (f2-f3).");
  }

  const cv::Size size = wrapped_phases[0].size();
  for (const auto& phase : wrapped_phases) {
    if (phase.size() != size || phase.type() != CV_64F) {
      throw std::invalid_argument(
          "All wrapped phase images must share identical dimensions and "
          "CV_64F type.");
    }
  }

  const bool use_modulation =
      validateOptionalMaps(modulations, size) && min_modulation > 0.0;

  const double R12 = static_cast<double>(f1) / static_cast<double>(f12);
  const double R123 = static_cast<double>(f12) / static_cast<double>(f123);

  cv::Mat result(size, CV_64F);

  for (int y = 0; y < size.height; ++y) {
    const double* p1_ptr = wrapped_phases[0].ptr<double>(y);
    const double* p2_ptr = wrapped_phases[1].ptr<double>(y);
    const double* p3_ptr = wrapped_phases[2].ptr<double>(y);

    const double* m1_ptr =
        use_modulation ? modulations[0].ptr<double>(y) : nullptr;
    const double* m2_ptr =
        use_modulation ? modulations[1].ptr<double>(y) : nullptr;
    const double* m3_ptr =
        use_modulation ? modulations[2].ptr<double>(y) : nullptr;

    double* out_ptr = result.ptr<double>(y);

    for (int x = 0; x < size.width; ++x) {
      if (use_modulation) {
        const double min_modulation_value =
            std::min({m1_ptr[x], m2_ptr[x], m3_ptr[x]});
        if (min_modulation_value < min_modulation) {
          out_ptr[x] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
      }

      const double p1 = p1_ptr[x];
      const double p2 = p2_ptr[x];
      const double p3 = p3_ptr[x];

      double p12 = p1 - p2;
      if (p12 < 0.0) p12 += kTwoPi;

      double p23 = p2 - p3;
      if (p23 < 0.0) p23 += kTwoPi;

      double p123 = p12 - p23;
      if (p123 < 0.0) p123 += kTwoPi;

      const double k12 = std::round((p123 * R123 - p12) / kTwoPi);
      const double abs12 = p12 + k12 * kTwoPi;
      const double k1 = std::round((abs12 * R12 - p1) / kTwoPi);
      out_ptr[x] = p1 + k1 * kTwoPi;
    }
  }

  return result;
}

cv::Mat PhaseProcessor::computeAbsolutePhase(
    const std::vector<cv::Mat>& wrapped_phases,
    const std::vector<int>& frequencies,
    const std::vector<cv::Mat>& modulations,
    const std::vector<cv::Mat>& fit_residual_ratios,
    const std::vector<cv::Mat>& saturation_fractions,
    const PhaseQualityOptions& quality_options, cv::Mat* out_confidence,
    PhaseQualitySummary* out_summary) {
  if (wrapped_phases.size() != 3 || frequencies.size() != 3) {
    throw std::invalid_argument(
        "Quality-aware unwrap expects exactly 3 frequencies.");
  }

  const cv::Size size = wrapped_phases[0].size();
  for (const auto& phase : wrapped_phases) {
    if (phase.empty() || phase.size() != size || phase.type() != CV_64F) {
      throw std::invalid_argument(
          "Wrapped phase maps must share size and CV_64F type.");
    }
  }

  if (!validateOptionalMaps(modulations, size) ||
      !validateOptionalMaps(fit_residual_ratios, size) ||
      !validateOptionalMaps(saturation_fractions, size)) {
    throw std::invalid_argument(
        "Quality-aware unwrap requires three modulation, fit-residual, and "
        "saturation maps.");
  }

  const int f1 = frequencies[0];
  const int f2 = frequencies[1];
  const int f3 = frequencies[2];
  const int f12 = f1 - f2;
  const int f23 = f2 - f3;
  const int f123 = f12 - f23;

  if (f12 <= 0 || f23 <= 0 || f123 <= 0) {
    throw std::invalid_argument("Invalid three-frequency combination.");
  }

  if (!(quality_options.min_modulation > 0.0) ||
      !(quality_options.max_fit_residual_ratio > 0.0) ||
      !(quality_options.max_frequency_consistency_rad > 0.0) ||
      !(quality_options.max_saturation_fraction > 0.0) ||
      !(quality_options.fusion_fit_residual_floor > 0.0)) {
    throw std::invalid_argument("Phase quality thresholds must be positive.");
  }

  const double R12 = static_cast<double>(f1) / static_cast<double>(f12);
  const double R123 = static_cast<double>(f12) / static_cast<double>(f123);

  cv::Mat result(size, CV_64F,
                 cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  cv::Mat confidence = cv::Mat::zeros(size, CV_64F);

  PhaseQualitySummary summary;
  summary.total_pixels = size.width * size.height;

  double confidence_sum = 0.0;
  double fusion_disagreement_sum = 0.0;
  std::vector<double> consistency_errors;
  consistency_errors.reserve(static_cast<std::size_t>(size.width) *
                             static_cast<std::size_t>(size.height) / 2);

  for (int y = 0; y < size.height; ++y) {
    const double* p1_ptr = wrapped_phases[0].ptr<double>(y);
    const double* p2_ptr = wrapped_phases[1].ptr<double>(y);
    const double* p3_ptr = wrapped_phases[2].ptr<double>(y);

    const double* m1_ptr = modulations[0].ptr<double>(y);
    const double* m2_ptr = modulations[1].ptr<double>(y);
    const double* m3_ptr = modulations[2].ptr<double>(y);

    const double* r1_ptr = fit_residual_ratios[0].ptr<double>(y);
    const double* r2_ptr = fit_residual_ratios[1].ptr<double>(y);
    const double* r3_ptr = fit_residual_ratios[2].ptr<double>(y);

    const double* s1_ptr = saturation_fractions[0].ptr<double>(y);
    const double* s2_ptr = saturation_fractions[1].ptr<double>(y);
    const double* s3_ptr = saturation_fractions[2].ptr<double>(y);

    double* out_ptr = result.ptr<double>(y);
    double* confidence_ptr = confidence.ptr<double>(y);

    for (int x = 0; x < size.width; ++x) {
      const double p1 = p1_ptr[x];
      const double p2 = p2_ptr[x];
      const double p3 = p3_ptr[x];

      if (!std::isfinite(p1) || !std::isfinite(p2) || !std::isfinite(p3)) {
        continue;
      }

      const double min_modulation_value =
          std::min({m1_ptr[x], m2_ptr[x], m3_ptr[x]});
      const double max_fit_ratio = std::max({r1_ptr[x], r2_ptr[x], r3_ptr[x]});
      const double max_saturation = std::max({s1_ptr[x], s2_ptr[x], s3_ptr[x]});

      const bool modulation_bad =
          !std::isfinite(min_modulation_value) ||
          min_modulation_value < quality_options.min_modulation;
      const bool fit_bad =
          !std::isfinite(max_fit_ratio) ||
          max_fit_ratio > quality_options.max_fit_residual_ratio;
      const bool saturation_bad =
          !std::isfinite(max_saturation) ||
          max_saturation > quality_options.max_saturation_fraction;

      if (modulation_bad) ++summary.rejected_modulation;
      if (fit_bad) ++summary.rejected_fit;
      if (saturation_bad) ++summary.rejected_saturation;

      if (modulation_bad || fit_bad || saturation_bad) {
        continue;
      }

      double p12 = p1 - p2;
      if (p12 < 0.0) p12 += kTwoPi;

      double p23 = p2 - p3;
      if (p23 < 0.0) p23 += kTwoPi;

      double p123 = p12 - p23;
      if (p123 < 0.0) p123 += kTwoPi;

      const double k12 = std::round((p123 * R123 - p12) / kTwoPi);
      const double abs12 = p12 + k12 * kTwoPi;
      const double k1 = std::round((abs12 * R12 - p1) / kTwoPi);
      const double absolute_phase_1 = p1 + k1 * kTwoPi;

      const double predicted_p2 =
          absolute_phase_1 * static_cast<double>(f2) / static_cast<double>(f1);
      const double predicted_p3 =
          absolute_phase_1 * static_cast<double>(f3) / static_cast<double>(f1);

      const double consistency_error =
          std::max(circularPhaseError(predicted_p2, p2),
                   circularPhaseError(predicted_p3, p3));

      if (std::isfinite(consistency_error)) {
        consistency_errors.push_back(consistency_error);
      }

      summary.max_frequency_consistency_rad =
          std::max(summary.max_frequency_consistency_rad, consistency_error);

      if (!std::isfinite(consistency_error) ||
          consistency_error > quality_options.max_frequency_consistency_rad) {
        ++summary.rejected_consistency;
        continue;
      }

      double final_absolute_phase = absolute_phase_1;
      double fusion_disagreement = 0.0;

      if (quality_options.enable_multifrequency_fusion) {
        const double expected_abs2 = absolute_phase_1 *
                                     static_cast<double>(f2) /
                                     static_cast<double>(f1);
        const double expected_abs3 = absolute_phase_1 *
                                     static_cast<double>(f3) /
                                     static_cast<double>(f1);

        const double k2 = std::round((expected_abs2 - p2) / kTwoPi);
        const double k3 = std::round((expected_abs3 - p3) / kTwoPi);

        const double absolute_phase_2 = p2 + k2 * kTwoPi;
        const double absolute_phase_3 = p3 + k3 * kTwoPi;

        final_absolute_phase = fuseUnwrappedPhases(
            absolute_phase_1, absolute_phase_2, absolute_phase_3, f1, f2, f3,
            r1_ptr[x], r2_ptr[x], r3_ptr[x], s1_ptr[x], s2_ptr[x], s3_ptr[x],
            quality_options, fusion_disagreement);

        fusion_disagreement_sum += fusion_disagreement;
        summary.max_fusion_disagreement_rad =
            std::max(summary.max_fusion_disagreement_rad, fusion_disagreement);
      }

      const double modulation_score = std::clamp(
          min_modulation_value / (2.0 * quality_options.min_modulation), 0.0,
          1.0);
      const double fit_score = std::clamp(
          1.0 - max_fit_ratio / quality_options.max_fit_residual_ratio, 0.0,
          1.0);
      const double saturation_score = std::clamp(
          1.0 - max_saturation / quality_options.max_saturation_fraction, 0.0,
          1.0);
      const double consistency_score =
          std::clamp(1.0 - consistency_error /
                               quality_options.max_frequency_consistency_rad,
                     0.0, 1.0);

      const double pixel_confidence =
          std::pow(std::max(0.0, modulation_score * fit_score *
                                     saturation_score * consistency_score),
                   0.25);

      out_ptr[x] = final_absolute_phase;
      confidence_ptr[x] = pixel_confidence;
      confidence_sum += pixel_confidence;
      ++summary.valid_pixels;
    }
  }

  if (summary.valid_pixels > 0) {
    summary.mean_confidence =
        confidence_sum / static_cast<double>(summary.valid_pixels);

    if (quality_options.enable_multifrequency_fusion) {
      summary.mean_fusion_disagreement_rad =
          fusion_disagreement_sum / static_cast<double>(summary.valid_pixels);
    }
  }

  summary.frequency_consistency_p95_rad = percentile(consistency_errors, 0.95);
  summary.frequency_consistency_p99_rad = percentile(consistency_errors, 0.99);

  if (out_confidence != nullptr) {
    *out_confidence = confidence;
  }
  if (out_summary != nullptr) {
    *out_summary = summary;
  }

  return result;
}

cv::Mat PhaseProcessor::filterPhaseNoise(const cv::Mat& phase,
                                         int kernel_size) {
  if (phase.empty() || phase.type() != CV_64F) {
    throw std::invalid_argument(
        "filterPhaseNoise expects a non-empty CV_64F phase map.");
  }

  if (kernel_size <= 1) {
    return phase.clone();
  }

  if (kernel_size % 2 == 0) {
    throw std::invalid_argument("Median phase filter kernel size must be odd.");
  }

  const int rows = phase.rows;
  const int cols = phase.cols;
  const int radius = kernel_size / 2;

  cv::Mat filtered(rows, cols, CV_64F,
                   cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));

  for (int y = 0; y < rows; ++y) {
    const double* source_row = phase.ptr<double>(y);
    double* destination_row = filtered.ptr<double>(y);

    for (int x = 0; x < cols; ++x) {
      if (!std::isfinite(source_row[x])) {
        continue;
      }

      values.clear();

      for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= rows) {
          continue;
        }

        const double* neighbor_row = phase.ptr<double>(ny);

        for (int dx = -radius; dx <= radius; ++dx) {
          const int nx = x + dx;
          if (nx < 0 || nx >= cols) {
            continue;
          }

          const double value = neighbor_row[nx];
          if (std::isfinite(value)) {
            values.push_back(value);
          }
        }
      }

      if (values.size() >= 3) {
        const std::size_t middle = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + middle, values.end());
        destination_row[x] = values[middle];
      } else {
        destination_row[x] = source_row[x];
      }
    }
  }

  return filtered;
}

cv::Mat PhaseProcessor::filterPhaseLocalPlane(
    const cv::Mat& phase, const cv::Mat& confidence, int kernel_size,
    int min_valid_neighbors, double robust_scale_rad, cv::Mat* out_confidence) {
  if (phase.empty() || confidence.empty() || phase.type() != CV_64F ||
      confidence.type() != CV_64F || phase.size() != confidence.size()) {
    throw std::invalid_argument(
        "Local-plane phase filtering requires matching CV_64F phase and "
        "confidence maps.");
  }

  if (kernel_size < 3 || kernel_size % 2 == 0) {
    throw std::invalid_argument(
        "Local-plane phase filter kernel size must be odd and >= 3.");
  }

  if (min_valid_neighbors < 3 ||
      min_valid_neighbors > kernel_size * kernel_size) {
    throw std::invalid_argument("Invalid local-plane minimum neighbor count.");
  }

  if (!(robust_scale_rad > 0.0)) {
    throw std::invalid_argument("Local-plane robust scale must be positive.");
  }

  const int radius = kernel_size / 2;
  const int full_support = kernel_size * kernel_size;

  cv::Mat filtered(phase.size(), CV_64F,
                   cv::Scalar(std::numeric_limits<double>::quiet_NaN()));
  cv::Mat filtered_confidence = cv::Mat::zeros(phase.size(), CV_64F);

  for (int y = 0; y < phase.rows; ++y) {
    const double* source_row = phase.ptr<double>(y);
    const double* source_confidence_row = confidence.ptr<double>(y);
    double* destination_row = filtered.ptr<double>(y);
    double* destination_confidence_row = filtered_confidence.ptr<double>(y);

    for (int x = 0; x < phase.cols; ++x) {
      if (!std::isfinite(source_row[x])) {
        continue;
      }

      const LocalPlaneFit initial = fitLocalPlane(
          phase, confidence, x, y, radius, robust_scale_rad, false, nullptr);

      if (!initial.valid || initial.valid_count < min_valid_neighbors) {
        destination_row[x] = source_row[x];
        destination_confidence_row[x] =
            std::clamp(source_confidence_row[x], 0.0, 1.0);
        continue;
      }

      const LocalPlaneFit robust = fitLocalPlane(
          phase, confidence, x, y, radius, robust_scale_rad, true, &initial);

      const LocalPlaneFit& fit =
          (robust.valid && robust.valid_count >= min_valid_neighbors) ? robust
                                                                      : initial;

      destination_row[x] = fit.center;

      const double support_score =
          std::sqrt(std::clamp(static_cast<double>(fit.valid_count) /
                                   static_cast<double>(full_support),
                               0.0, 1.0));
      const double fit_score =
          1.0 / std::sqrt(1.0 + (fit.weighted_rmse / robust_scale_rad) *
                                    (fit.weighted_rmse / robust_scale_rad));

      destination_confidence_row[x] =
          std::clamp(fit.mean_confidence * support_score * fit_score, 0.0, 1.0);
    }
  }

  if (out_confidence != nullptr) {
    *out_confidence = filtered_confidence;
  }

  return filtered;
}

bool PhaseProcessor::savePhaseEXR(const cv::Mat& phase,
                                  const std::string& filename, bool compress) {
  if (phase.empty()) {
    return false;
  }

  cv::Mat output;
  phase.convertTo(output, CV_32F);

  std::vector<int> parameters;
  if (!compress) {
    parameters = {cv::IMWRITE_EXR_TYPE, cv::IMWRITE_EXR_TYPE_FLOAT,
                  cv::IMWRITE_EXR_COMPRESSION, cv::IMWRITE_EXR_COMPRESSION_NO};
  }

  return cv::imwrite(filename, output, parameters);
}

}  // namespace msm3d

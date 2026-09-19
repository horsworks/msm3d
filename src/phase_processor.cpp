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
      !(quality_options.max_saturation_fraction > 0.0)) {
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

      const double final_absolute_phase = absolute_phase_1;

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

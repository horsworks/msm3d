#include "msm3d/phase_processor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace msm3d {

cv::Mat PhaseProcessor::computeWrappedPhase(const std::vector<cv::Mat>& images,
                                            cv::Mat* out_modulation) {
  if (images.empty()) {
    throw std::invalid_argument("Input fringe images list is empty.");
  }

  const cv::Size img_size = images[0].size();
  cv::Mat sin_sum = cv::Mat::zeros(img_size, CV_64F);
  cv::Mat cos_sum = cv::Mat::zeros(img_size, CV_64F);

  const std::size_t n = images.size();
  const double two_pi_over_n = 2.0 * CV_PI / static_cast<double>(n);

  cv::Mat gray_f64;

  for (std::size_t i = 0; i < n; ++i) {
    if (images[i].size() != img_size) {
      throw std::invalid_argument(
          "All fringe images must share the same resolution.");
    }

    if (images[i].channels() == 1) {
      images[i].convertTo(gray_f64, CV_64F);
    } else {
      cv::Mat gray;
      cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
      gray.convertTo(gray_f64, CV_64F);
    }

    const double delta = two_pi_over_n * static_cast<double>(i);
    const double sin_val = std::sin(delta);
    const double cos_val = std::cos(delta);

    cv::scaleAdd(gray_f64, sin_val, sin_sum, sin_sum);
    cv::scaleAdd(gray_f64, cos_val, cos_sum, cos_sum);
  }

  cv::Mat phase;
  cv::Mat neg_sin_sum = -sin_sum;
  cv::phase(cos_sum, neg_sin_sum, phase, false);

  if (out_modulation != nullptr) {
    cv::Mat mag;
    cv::magnitude(cos_sum, sin_sum, mag);
    *out_modulation = mag * (2.0 / static_cast<double>(n));
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
        "Invalid frequency combination: requires f1 > f2 > f3 and (f1-f2) > "
        "(f2-f3).");
  }

  const cv::Size size = wrapped_phases[0].size();
  for (const auto& mat : wrapped_phases) {
    if (mat.size() != size || mat.type() != CV_64F) {
      throw std::invalid_argument(
          "All wrapped phase images must share identical dimensions and CV_64F "
          "type.");
    }
  }

  const bool use_modulation =
      (!modulations.empty() && modulations.size() == 3 && min_modulation > 0.0);

  const double R12 = static_cast<double>(f1) / static_cast<double>(f12);
  const double R123 = static_cast<double>(f12) / static_cast<double>(f123);
  constexpr double kTwoPi = 2.0 * CV_PI;

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
      // 调制度门限截断：低信噪比暗区与黑圆点直接赋予 NaN
      if (use_modulation) {
        const double min_mod = std::min({m1_ptr[x], m2_ptr[x], m3_ptr[x]});
        if (min_mod < min_modulation) {
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

cv::Mat PhaseProcessor::filterPhaseNoise(const cv::Mat& phase,
                                         int kernel_size) {
  if (kernel_size <= 1) {
    return phase.clone();
  }

  const int rows = phase.rows;
  const int cols = phase.cols;
  const int r = kernel_size / 2;
  cv::Mat filtered = cv::Mat(
      rows, cols, CV_64F, cv::Scalar(std::numeric_limits<double>::quiet_NaN()));

  std::vector<double> vals;
  vals.reserve((2 * r + 1) * (2 * r + 1));

  for (int y = 0; y < rows; ++y) {
    const double* src_row = phase.ptr<double>(y);
    double* dst_row = filtered.ptr<double>(y);

    for (int x = 0; x < cols; ++x) {
      if (!std::isfinite(src_row[x])) {
        continue;
      }

      vals.clear();
      for (int dy = -r; dy <= r; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= rows) continue;
        const double* n_ptr = phase.ptr<double>(ny);
        for (int dx = -r; dx <= r; ++dx) {
          const int nx = x + dx;
          if (nx < 0 || nx >= cols) continue;
          const double v = n_ptr[nx];
          if (std::isfinite(v)) {
            vals.push_back(v);
          }
        }
      }

      if (vals.size() >= 3) {
        const std::size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
        dst_row[x] = vals[mid];
      } else {
        dst_row[x] = src_row[x];
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

  cv::Mat out;
  phase.convertTo(out, CV_32F);

  std::vector<int> params;
  if (!compress) {
    params = {cv::IMWRITE_EXR_TYPE, cv::IMWRITE_EXR_TYPE_FLOAT,
              cv::IMWRITE_EXR_COMPRESSION, cv::IMWRITE_EXR_COMPRESSION_NO};
  }

  return cv::imwrite(filename, out, params);
}

}  // namespace msm3d
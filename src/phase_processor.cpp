#include "msm3d/phase_processor.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

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

  // 1. 核心相位解算: [0, 2*pi)
  cv::Mat phase;
  cv::phase(cos_sum, sin_sum, phase, false);

  // 2. 仅在显式请求时才计算调制度，避免冗余开方与矩阵分配
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
  if (wrapped_phases.size() != 3 || frequencies.size() != 3) {
    throw std::invalid_argument(
        "Multi-frequency unwrap expects exactly 3 wrapped phases and 3 "
        "frequencies.");
  }

  if (frequencies[0] <= frequencies[1] || frequencies[1] <= frequencies[2]) {
    throw std::invalid_argument(
        "Frequencies must be in strictly descending order (e.g., 70, 64, 59).");
  }

  const cv::Size size = wrapped_phases[0].size();
  for (const auto& mat : wrapped_phases) {
    if (mat.size() != size || mat.type() != CV_64F) {
      throw std::invalid_argument(
          "All wrapped phase images must share identical dimensions and CV_64F "
          "type.");
    }
  }

  const double f1 = static_cast<double>(frequencies[0]);
  const double f2 = static_cast<double>(frequencies[1]);
  const double f3 = static_cast<double>(frequencies[2]);

  const double T1 = 1.0 / f1;
  const double T2 = 1.0 / f2;
  const double T3 = 1.0 / f3;

  const double T12 = (T1 * T2) / (T2 - T1);
  const double T23 = (T2 * T3) / (T3 - T2);
  const double T123 = (T12 * T23) / (T23 - T12);

  const double R12 = T12 / T1;
  const double R123 = T123 / T12;
  constexpr double kTwoPi = 2.0 * CV_PI;

  cv::Mat result(size, CV_64F);

  for (int y = 0; y < size.height; ++y) {
    const double* p1_ptr = wrapped_phases[0].ptr<double>(y);
    const double* p2_ptr = wrapped_phases[1].ptr<double>(y);
    const double* p3_ptr = wrapped_phases[2].ptr<double>(y);
    double* out_ptr = result.ptr<double>(y);

    for (int x = 0; x < size.width; ++x) {
      const double p1 = p1_ptr[x];
      const double p2 = p2_ptr[x];
      const double p3 = p3_ptr[x];

      double p12 = p1 - p2;
      if (p12 < 0.0) p12 += kTwoPi;

      double p23 = p2 - p3;
      if (p23 < 0.0) p23 += kTwoPi;

      double p123 = p12 - p23;
      if (p123 < 0.0) p123 += kTwoPi;

      const double abs12 =
          p12 + std::round((p123 * R123 - p12) / kTwoPi) * kTwoPi;
      const double k = std::round((abs12 * R12 - p1) / kTwoPi);

      out_ptr[x] = p1 + k * kTwoPi;
    }
  }

  return result;
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
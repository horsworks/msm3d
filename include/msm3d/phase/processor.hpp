#pragma once

#include "msm3d/phase/quality.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

class PhaseProcessor {
 public:
  static cv::Mat computeWrappedPhase(
      const std::vector<cv::Mat>& images, cv::Mat* out_modulation = nullptr,
      cv::Mat* out_fit_residual_ratio = nullptr,
      cv::Mat* out_saturation_fraction = nullptr);

  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies);

  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies,
      const std::vector<cv::Mat>& modulations, double min_modulation = 8.0);

  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies,
      const std::vector<cv::Mat>& modulations,
      const std::vector<cv::Mat>& fit_residual_ratios,
      const std::vector<cv::Mat>& saturation_fractions,
      const PhaseQualityOptions& quality_options,
      cv::Mat* out_confidence = nullptr,
      PhaseQualitySummary* out_summary = nullptr);

  static cv::Mat filterPhaseNoise(const cv::Mat& phase, int kernel_size = 3);

  static bool savePhaseEXR(const cv::Mat& phase, const std::string& filename,
                           bool compress = false);
};

}  // namespace msm3d

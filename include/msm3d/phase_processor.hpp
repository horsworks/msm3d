#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

struct PhaseQualityOptions {
  double min_modulation = 8.0;
  double max_fit_residual_ratio = 0.35;
  double max_frequency_consistency_rad = 0.25;
  double max_saturation_fraction = 0.25;

  bool enable_multifrequency_fusion = true;
  double fusion_fit_residual_floor = 0.03;
};

struct PhaseQualitySummary {
  int total_pixels = 0;
  int valid_pixels = 0;
  int rejected_modulation = 0;
  int rejected_fit = 0;
  int rejected_saturation = 0;
  int rejected_consistency = 0;

  double mean_confidence = 0.0;
  double max_frequency_consistency_rad = 0.0;
  double frequency_consistency_p95_rad = 0.0;
  double frequency_consistency_p99_rad = 0.0;

  double mean_fusion_disagreement_rad = 0.0;
  double max_fusion_disagreement_rad = 0.0;
};

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

  static cv::Mat filterPhaseLocalPlane(const cv::Mat& phase,
                                       const cv::Mat& confidence,
                                       int kernel_size = 3,
                                       int min_valid_neighbors = 5,
                                       double robust_scale_rad = 0.10,
                                       cv::Mat* out_confidence = nullptr);

  static bool savePhaseEXR(const cv::Mat& phase, const std::string& filename,
                           bool compress = false);
};

}  // namespace msm3d

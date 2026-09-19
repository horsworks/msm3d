#pragma once

namespace msm3d {

struct PhaseQualityOptions {
  double min_modulation = 8.0;
  double max_fit_residual_ratio = 0.35;
  double max_frequency_consistency_rad = 0.25;
  double max_saturation_fraction = 0.25;
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
};

}  // namespace msm3d

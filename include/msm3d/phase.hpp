#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace msm3d {

struct PhaseResult {
  cv::Mat absolute_phase;
  cv::Mat confidence;
  cv::Mat valid_mask;
};

class PhaseProcessor {
 public:
  PhaseResult compute(const std::vector<cv::Mat>& images) const;
};

}  // namespace msm3d

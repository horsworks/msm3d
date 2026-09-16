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
  PhaseResult computePhase(const std::vector<cv::Mat>& images) const;

 private:
  cv::Mat computeConfidence(const cv::Mat& modulation) const;
};

}  // namespace msm3d

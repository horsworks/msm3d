#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace msm3d {

class PhaseProcessor {
 public:
  cv::Mat computeWrappedPhase(const std::vector<cv::Mat>& images) const;

  cv::Mat computeAbsolutePhase(const std::vector<cv::Mat>& wrapped_phases,
                               const std::vector<int>& frequencies) const;

 private:
  cv::Mat unwrapPair(const cv::Mat& phase1, const cv::Mat& phase2,
                     int frequency1, int frequency2) const;
};

}  // namespace msm3d

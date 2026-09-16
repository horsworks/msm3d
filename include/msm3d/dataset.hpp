#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace msm3d {

struct CalibrationSample {
  std::vector<cv::Mat> fringe_images;
  cv::Mat phase;
};

}  // namespace msm3d

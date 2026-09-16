#include <cassert>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "msm3d/phase.hpp"

int main() {
  std::vector<cv::Mat> images;

  for (int i = 0; i < 4; ++i) {
    images.emplace_back(cv::Mat::ones(10, 10, CV_8U) * i);
  }

  msm3d::PhaseProcessor processor;

  auto result = processor.computePhase(images);

  assert(!result.absolute_phase.empty());

  std::cout << "phase test passed\n";

  return 0;
}

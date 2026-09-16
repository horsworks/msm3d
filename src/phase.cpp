#include "msm3d/phase.hpp"

#include <cmath>

namespace msm3d {

cv::Mat PhaseProcessor::computeWrappedPhase(
    const std::vector<cv::Mat>& images) const {
  CV_Assert(!images.empty());

  cv::Mat numerator = cv::Mat::zeros(images[0].size(), CV_64F);

  cv::Mat denominator = cv::Mat::zeros(images[0].size(), CV_64F);

  int count = static_cast<int>(images.size());

  for (int k = 0; k < count; ++k) {
    double delta = 2.0 * CV_PI * k / count;

    cv::Mat image_double;

    images[k].convertTo(image_double, CV_64F);

    numerator += image_double * std::sin(delta);

    denominator += image_double * std::cos(delta);
  }

  cv::Mat phase;

  cv::phase(denominator, numerator, phase);

  return phase;
}

cv::Mat PhaseProcessor::computeAbsolutePhase(
    const std::vector<cv::Mat>& wrapped_phases,
    const std::vector<int>& frequencies) const {
  CV_Assert(wrapped_phases.size() == frequencies.size());

  if (wrapped_phases.size() < 2) {
    return wrapped_phases.front().clone();
  }

  return unwrapPair(wrapped_phases[0], wrapped_phases[1], frequencies[0],
                    frequencies[1]);
}

cv::Mat PhaseProcessor::unwrapPair(const cv::Mat& phase1, const cv::Mat& phase2,
                                   int frequency1, int frequency2) const {
  cv::Mat result = cv::Mat::zeros(phase1.size(), CV_64F);

  for (int y = 0; y < phase1.rows; ++y) {
    for (int x = 0; x < phase1.cols; ++x) {
      double p1 = phase1.at<double>(y, x);

      double p2 = phase2.at<double>(y, x);

      double k = std::round((frequency2 * p1 - frequency1 * p2) /
                            (2.0 * CV_PI * (frequency1 - frequency2)));

      result.at<double>(y, x) = p1 + 2.0 * CV_PI * k;
    }
  }

  return result;
}

}  // namespace msm3d

#include "msm3d/phase.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace msm3d {

PhaseResult PhaseProcessor::computePhase(
    const std::vector<cv::Mat>& images) const {
  PhaseResult result;

  if (images.empty()) {
    return result;
  }

  const int rows = images[0].rows;
  const int cols = images[0].cols;

  cv::Mat numerator = cv::Mat::zeros(rows, cols, CV_64F);

  cv::Mat denominator = cv::Mat::zeros(rows, cols, CV_64F);

  cv::Mat modulation = cv::Mat::zeros(rows, cols, CV_64F);

  const int n = static_cast<int>(images.size());

  for (int k = 0; k < n; ++k) {
    cv::Mat image_double;

    images[k].convertTo(image_double, CV_64F);

    const double delta = 2.0 * CV_PI * k / n;

    numerator += image_double * (-std::sin(delta));

    denominator += image_double * std::cos(delta);
  }

  cv::phase(denominator, numerator, result.absolute_phase, false);

  cv::Mat abs_num;
  cv::Mat abs_den;

  cv::absdiff(numerator, cv::Scalar(0), abs_num);

  cv::absdiff(denominator, cv::Scalar(0), abs_den);

  modulation = abs_num + abs_den;

  result.confidence = computeConfidence(modulation);

  result.valid_mask = result.confidence > 5e-2;

  return result;
}

cv::Mat PhaseProcessor::computeConfidence(const cv::Mat& modulation) const {
  cv::Mat normalized;

  cv::normalize(modulation, normalized, 0.0, 1.0, cv::NORM_MINMAX);

  return normalized;
}

}  // namespace msm3d

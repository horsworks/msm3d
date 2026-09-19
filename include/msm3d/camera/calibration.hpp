#pragma once

#include "msm3d/camera/types.hpp"

namespace msm3d {

std::vector<cv::Point3f> generateCalibrationObjectPoints(
    const CalibrationBoard& board);

CalibrationDetectionResult detectCalibrationPoints(
    const cv::Mat& image, const CalibrationBoard& board,
    const CircleDetectorParameters& detector_params);

CameraCalibrationResult calibrateCamera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const cv::Size& image_size, const CameraCalibrationOptions& options);

}  // namespace msm3d

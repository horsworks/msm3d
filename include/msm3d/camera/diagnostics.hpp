#pragma once

#include "msm3d/camera/types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace msm3d {

void printCameraIntrinsicStdDeviations(const cv::Mat& stddev);

void printWorstCameraViews(const CameraCalibrationResult& result,
                           const std::vector<std::string>& valid_view_names,
                           std::size_t max_count = 10);

void saveCameraResidualDiagnostics(
    const std::filesystem::path& diagnostics_dir,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::string>& valid_view_names,
    const CameraCalibrationResult& result, const CalibrationBoard& board,
    const cv::Size& image_size);

}  // namespace msm3d

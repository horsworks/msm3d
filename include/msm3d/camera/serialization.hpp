#pragma once

#include "msm3d/camera/types.hpp"

#include <string>
#include <vector>

namespace msm3d {

bool saveCameraCalibrationResult(
    const std::string& file_path, const CameraCalibrationResult& result,
    const cv::Size& image_size,
    const std::vector<std::string>& valid_view_names = {});

CameraCalibrationResult loadCameraCalibrationResult(
    const std::string& result_path);

}  // namespace msm3d

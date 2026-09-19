#pragma once

#include "msm3d/camera/types.hpp"
#include "msm3d/msm/model.hpp"
#include "msm3d/msm/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace msm3d {

MsmCalibrationResult calibrateMsm(const MsmCalibrationConfig& config,
                                  const CameraCalibrationResult& camera_calib,
                                  const std::vector<cv::Mat>& all_phase_maps);

}  // namespace msm3d

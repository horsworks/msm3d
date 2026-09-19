#pragma once

#include "msm3d/msm/model.hpp"

#include <string>

namespace msm3d {

bool saveMsmCalibrationResult(const std::string& file_path,
                              const MsmCalibrationResult& result);

MsmCalibrationResult loadMsmCalibrationResult(const std::string& file_path);

}  // namespace msm3d

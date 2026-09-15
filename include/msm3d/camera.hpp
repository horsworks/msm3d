#pragma once

#include <Eigen/Core>

#include "msm3d/geometry.hpp"

namespace msm3d {

struct CameraModel {
  Eigen::Matrix3d intrinsic;

  Eigen::Matrix<double, 5, 1> distortion;
};

Ray3D pixelToRay(const CameraModel& camera, double u, double v);

}  // namespace msm3d
#pragma once

#include <Eigen/Core>

#include <opencv2/core.hpp>

#include "msm3d/geometry.hpp"

namespace msm3d {

struct CameraModel {
  Eigen::Matrix3d intrinsic;
  Eigen::VectorXd distortion;
};

CameraModel convertCameraModel(const cv::Mat& camera_matrix,
                               const cv::Mat& distortion);

Ray3D pixelToRay(const CameraModel& camera, double u, double v);

}  // namespace msm3d

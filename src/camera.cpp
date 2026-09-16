#include "msm3d/camera.hpp"

namespace msm3d {

CameraModel convertCameraModel(const cv::Mat& camera_matrix,
                               const cv::Mat& distortion) {
  CameraModel model;

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      model.intrinsic(r, c) = camera_matrix.at<double>(r, c);
    }
  }

  model.distortion.resize(distortion.total());

  for (int i = 0; i < static_cast<int>(distortion.total()); ++i) {
    model.distortion(i) = distortion.at<double>(i);
  }

  return model;
}

Ray3D pixelToRay(const CameraModel& camera, double u, double v) {
  Eigen::Vector3d direction;

  direction << (u - camera.intrinsic(0, 2)) / camera.intrinsic(0, 0),
      (v - camera.intrinsic(1, 2)) / camera.intrinsic(1, 1), 1.0;

  return Ray3D(Eigen::Vector3d::Zero(), direction);
}

}  // namespace msm3d

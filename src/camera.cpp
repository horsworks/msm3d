#include "msm3d/camera.hpp"

namespace msm3d {

Ray3D pixelToRay(const CameraModel& camera, double u, double v) {
  Eigen::Vector3d direction;

  direction << (u - camera.intrinsic(0, 2)) / camera.intrinsic(0, 0),
      (v - camera.intrinsic(1, 2)) / camera.intrinsic(1, 1), 1.0;

  return Ray3D(Eigen::Vector3d::Zero(), direction);
}

}  // namespace msm3d

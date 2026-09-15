#include "msm3d/camera.hpp"

namespace msm3d {

Ray3D pixelToRay(const CameraModel&, double, double) {
  return Ray3D(Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, 1));
}

}  // namespace msm3d
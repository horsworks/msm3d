#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace msm3d {

struct Ray3D {
  Eigen::Vector3d origin;
  Eigen::Vector3d direction;

  Ray3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction);
};

struct Plane3D {
  Eigen::Vector3d normal;
  double d;

  Plane3D(const Eigen::Vector3d& normal, double d);
};

bool intersectRayPlane(const Ray3D& ray, const Plane3D& plane,
                       Eigen::Vector3d& point);

}  // namespace msm3d

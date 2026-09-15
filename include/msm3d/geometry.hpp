#pragma once

#include <Eigen/Core>

namespace msm3d {

struct Ray3D {
  Eigen::Vector3d origin;

  Eigen::Vector3d direction;

  Ray3D(const Eigen::Vector3d& o, const Eigen::Vector3d& d);
};

struct Plane3D {
  Eigen::Vector3d normal;

  double d;

  Plane3D(const Eigen::Vector3d& n, double offset);
};

struct Transform3D {
  Eigen::Matrix3d rotation;

  Eigen::Vector3d translation;

  Eigen::Vector3d transformPoint(const Eigen::Vector3d& point) const;

  Transform3D inverse() const;

  Transform3D operator*(const Transform3D& rhs) const;
};

bool intersect(const Ray3D& ray, const Plane3D& plane, Eigen::Vector3d& point);

}  // namespace msm3d
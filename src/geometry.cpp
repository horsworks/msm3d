#include "msm3d/geometry.hpp"

#include <cmath>

namespace msm3d {

Ray3D::Ray3D(const Eigen::Vector3d& o, const Eigen::Vector3d& d)
    : origin(o), direction(d.normalized()) {}

Plane3D::Plane3D(const Eigen::Vector3d& n, double offset)
    : normal(n.normalized()), d(offset) {}

Eigen::Vector3d Transform3D::transformPoint(
    const Eigen::Vector3d& point) const {
  return rotation * point + translation;
}

Transform3D Transform3D::inverse() const {
  Transform3D result;

  result.rotation = rotation.transpose();

  result.translation = -result.rotation * translation;

  return result;
}

Transform3D Transform3D::operator*(const Transform3D& rhs) const {
  Transform3D result;

  result.rotation = rotation * rhs.rotation;

  result.translation = rotation * rhs.translation + translation;

  return result;
}

bool intersect(const Ray3D& ray, const Plane3D& plane, Eigen::Vector3d& point) {
  constexpr double eps = 1e-12;

  const double denominator = plane.normal.dot(ray.direction);

  if (std::abs(denominator) < eps) {
    return false;
  }

  const double t = -(plane.normal.dot(ray.origin) + plane.d) / denominator;

  point = ray.origin + t * ray.direction;

  return true;
}

}  // namespace msm3d
#include "msm3d/geometry.hpp"

#include <cmath>

namespace msm3d {

Ray3D::Ray3D(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction)
    : origin(origin), direction(direction.normalized()) {}

Plane3D::Plane3D(const Eigen::Vector3d& normal, double d)
    : normal(normal.normalized()), d(d) {}

bool intersectRayPlane(const Ray3D& ray, const Plane3D& plane,
                       Eigen::Vector3d& point) {
  constexpr double kEpsilon = 1e-12;

  const double denom = plane.normal.dot(ray.direction);

  if (std::abs(denom) < kEpsilon) {
    return false;
  }

  const double t = -(plane.normal.dot(ray.origin) + plane.d) / denom;

  point = ray.origin + t * ray.direction;
  return true;
}

}  // namespace msm3d

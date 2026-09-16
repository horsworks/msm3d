#include "msm3d/geometry.hpp"

#include <cmath>

namespace msm3d {

double Plane::distance(const Eigen::Vector3d& point) const {
  return normal.dot(point) + d;
}

bool intersect(const Ray& ray, const Plane& plane, Eigen::Vector3d& point) {
  const double denom = plane.normal.dot(ray.direction);

  if (std::abs(denom) < 1e-12) {
    return false;
  }

  double lambda = -(plane.normal.dot(ray.origin) + plane.d) / denom;

  point = ray.origin + lambda * ray.direction;

  return true;
}

Eigen::Vector3d rodrigues(const Eigen::Vector3d& vector,
                          const Eigen::Vector3d& axis, double angle) {
  Eigen::Vector3d w = axis.normalized();

  return vector * std::cos(angle) + w.cross(vector) * std::sin(angle) +
         w * (w.dot(vector)) * (1.0 - std::cos(angle));
}

}  // namespace msm3d

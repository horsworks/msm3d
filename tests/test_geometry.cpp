#include <Eigen/Geometry>
#include <cassert>
#include <iostream>

#include "msm3d/geometry.hpp"

using namespace msm3d;

void testRayPlaneIntersection() {
  Ray3D ray(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 1));

  Plane3D plane(Eigen::Vector3d(0, 0, 1), -10);

  Eigen::Vector3d point;

  bool result = intersect(ray, plane, point);

  assert(result);

  assert(std::abs(point.z() - 10.0) < 1e-9);
}

void testParallelRay() {
  Ray3D ray(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0));

  Plane3D plane(Eigen::Vector3d(0, 0, 1), -10);

  Eigen::Vector3d point;

  bool result = intersect(ray, plane, point);

  assert(!result);
}

void testTransformInverse() {
  Transform3D T;

  T.rotation =
      Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  T.translation = Eigen::Vector3d(1, 2, 3);

  Eigen::Vector3d p(3, 4, 5);

  Eigen::Vector3d result = T.inverse().transformPoint(T.transformPoint(p));

  assert((result - p).norm() < 1e-9);
}

int main() {
  testRayPlaneIntersection();

  testParallelRay();

  testTransformInverse();

  std::cout << "Geometry tests passed\n";

  return 0;
}
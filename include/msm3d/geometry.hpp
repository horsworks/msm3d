#pragma once

#include <Eigen/Core>

namespace msm3d
{

struct Plane
{
    Eigen::Vector3d normal;
    double d = 0.0;

    double distance(
        const Eigen::Vector3d& point) const;
};

struct Ray
{
    Eigen::Vector3d origin;
    Eigen::Vector3d direction;
};

bool intersect(
    const Ray& ray,
    const Plane& plane,
    Eigen::Vector3d& point);

Eigen::Vector3d rodrigues(
    const Eigen::Vector3d& vector,
    const Eigen::Vector3d& axis,
    double angle);

}

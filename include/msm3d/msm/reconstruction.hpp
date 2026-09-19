#pragma once

#include "msm3d/msm/model.hpp"
#include "msm3d/msm/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace msm3d {

BoardPlane computeBoardPlane(const cv::Mat& rvec_or_R, const cv::Mat& t_mm);

std::vector<cv::Vec3d> projectSubpixelsToBoard(
    const std::vector<cv::Point2d>& subpixels, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs, const BoardPlane& board_plane,
    double min_depth_mm = 50.0, double max_depth_mm = 2000.0);

double evaluateMsmReconstruction(const MsmCalibrationResult& result,
                                 const cv::Mat& phase_map,
                                 const cv::Mat& camera_matrix,
                                 const cv::Mat& dist_coeffs,
                                 const BoardPlane& board_plane, double min_psi,
                                 double max_psi, int pixel_stride = 4);

}  // namespace msm3d

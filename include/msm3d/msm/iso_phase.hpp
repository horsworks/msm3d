#pragma once

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace msm3d {

std::pair<double, double> autoDetectValidPhaseRange(
    const std::vector<cv::Mat>& phase_maps, int min_covisible_poses = 3);

std::vector<cv::Point2d> extractIsoPhaseSubpixels(
    const cv::Mat& phase_map, double target_psi,
    const cv::Mat& mask = cv::Mat(), double grad_threshold = 1e-3,
    int fit_half_window = 2, double max_grad_threshold = 0.5,
    double max_lateral_jump_px = 5.0);

}  // namespace msm3d

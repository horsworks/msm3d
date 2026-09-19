#include "msm3d/msm_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace {

std::string inferConfidencePath(const std::string& phase_path) {
  const std::filesystem::path path(phase_path);
  const std::string name = path.filename().string();
  const std::string prefix = "pose_";

  if (name.rfind(prefix, 0) == 0) {
    return (path.parent_path() / ("confidence_" + name)).string();
  }

  return {};
}

cv::Mat finiteMask(const cv::Mat& values) {
  cv::Mat mask;
  cv::compare(values, values, mask, cv::CMP_EQ);
  return mask;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string phase_path =
      (argc > 1) ? argv[1] : "./output/phase/pose_5.exr";
  const std::string confidence_path =
      (argc > 2) ? argv[2] : inferConfidencePath(phase_path);
  const std::string out_dir = "./output/debug";

  if (!std::filesystem::exists(phase_path)) {
    std::cerr << "Phase map not found: " << phase_path << std::endl;
    return -1;
  }

  cv::Mat phase_raw = cv::imread(phase_path, cv::IMREAD_UNCHANGED);
  if (phase_raw.empty()) {
    std::cerr << "Failed to read: " << phase_path << std::endl;
    return -1;
  }

  cv::Mat phase_f64;
  if (phase_raw.channels() > 1) {
    cv::extractChannel(phase_raw, phase_f64, 0);
  } else {
    phase_f64 = phase_raw;
  }
  phase_f64.convertTo(phase_f64, CV_64F);

  const cv::Mat valid_mask = finiteMask(phase_f64);
  const int valid_pixels = cv::countNonZero(valid_mask);
  const int total_pixels = phase_f64.rows * phase_f64.cols;

  std::cout << "Image size: " << phase_f64.cols << " x " << phase_f64.rows
            << std::endl;
  std::cout << "Finite phase pixels: " << valid_pixels << " / " << total_pixels
            << " (" << std::fixed << std::setprecision(2)
            << 100.0 * valid_pixels / std::max(total_pixels, 1) << "%)"
            << std::endl;

  // Center-row phase profile.
  const int cols = phase_f64.cols;
  const int rows = phase_f64.rows;
  const int mid_y = rows / 2;
  const double* mid_row = phase_f64.ptr<double>(mid_y);

  const int plot_w = 1000;
  const int plot_h = 500;
  cv::Mat plot_img(plot_h, plot_w, CV_8UC3, cv::Scalar(245, 245, 245));

  double min_phase = 0.0;
  double max_phase = 1.0;
  cv::minMaxLoc(phase_f64, &min_phase, &max_phase, nullptr, nullptr,
                valid_mask);
  if (!(max_phase > min_phase)) {
    max_phase = min_phase + 1.0;
  }

  auto to_screen = [&](int image_x, double phase) -> cv::Point {
    const int sx = static_cast<int>(static_cast<double>(image_x) /
                                        std::max(cols - 1, 1) * (plot_w - 100) +
                                    60);
    const double t =
        std::clamp((phase - min_phase) / (max_phase - min_phase), 0.0, 1.0);
    const int sy = static_cast<int>((plot_h - 60) - t * (plot_h - 100));
    return cv::Point(sx, sy);
  };

  cv::line(plot_img, cv::Point(60, plot_h - 60),
           cv::Point(plot_w - 40, plot_h - 60), cv::Scalar(0, 0, 0), 2);
  cv::line(plot_img, cv::Point(60, 40), cv::Point(60, plot_h - 60),
           cv::Scalar(0, 0, 0), 2);

  for (int x = 1; x < cols; ++x) {
    if (!std::isfinite(mid_row[x - 1]) || !std::isfinite(mid_row[x])) {
      continue;
    }
    cv::line(plot_img, to_screen(x - 1, mid_row[x - 1]),
             to_screen(x, mid_row[x]), cv::Scalar(200, 50, 0), 2);
  }

  // Full-field phase visualization with invalid pixels shown in black.
  cv::Mat phase_for_vis = phase_f64.clone();
  phase_for_vis.setTo(min_phase, ~valid_mask);
  cv::Mat normalized;
  phase_for_vis.convertTo(normalized, CV_64F, 255.0 / (max_phase - min_phase),
                          -255.0 * min_phase / (max_phase - min_phase));
  cv::Mat gray8;
  normalized.convertTo(gray8, CV_8U);
  cv::Mat phase_vis;
  cv::applyColorMap(gray8, phase_vis, cv::COLORMAP_VIRIDIS);
  phase_vis.setTo(cv::Scalar(0, 0, 0), ~valid_mask);

  std::cout << "\n--- Iso-phase contours ---" << std::endl;
  for (double target_psi = 30.0; target_psi <= 165.0; target_psi += 30.0) {
    const auto subpixels =
        msm3d::extractIsoPhaseSubpixels(phase_f64, target_psi);
    if (subpixels.empty()) {
      continue;
    }

    std::cout << "  psi=" << std::setw(6) << target_psi
              << ", points=" << subpixels.size() << std::endl;

    for (std::size_t i = 0; i < subpixels.size(); ++i) {
      cv::circle(phase_vis, subpixels[i], 2, cv::Scalar(255, 255, 255), -1);
      if (i > 0 && std::abs(subpixels[i].y - subpixels[i - 1].y) <= 3.0) {
        cv::line(phase_vis, subpixels[i - 1], subpixels[i],
                 cv::Scalar(255, 255, 255), 2);
      }
    }
  }

  std::filesystem::create_directories(out_dir);
  cv::imwrite(out_dir + "/phase_profile_curve.png", plot_img);
  cv::imwrite(out_dir + "/phase_contours_overview.png", phase_vis);

  if (!confidence_path.empty() && std::filesystem::exists(confidence_path)) {
    cv::Mat confidence = cv::imread(confidence_path, cv::IMREAD_UNCHANGED);
    if (!confidence.empty()) {
      if (confidence.channels() > 1) {
        cv::extractChannel(confidence, confidence, 0);
      }
      confidence.convertTo(confidence, CV_64F);

      cv::Mat confidence8;
      confidence.convertTo(confidence8, CV_8U, 255.0);
      cv::Mat confidence_vis;
      cv::applyColorMap(confidence8, confidence_vis, cv::COLORMAP_TURBO);
      confidence_vis.setTo(cv::Scalar(0, 0, 0), ~valid_mask);
      cv::imwrite(out_dir + "/phase_confidence_overview.png", confidence_vis);

      cv::Scalar mean_confidence = cv::mean(confidence, valid_mask);
      std::cout << "Confidence map: " << confidence_path << std::endl;
      std::cout << "Mean confidence over valid phase pixels: "
                << mean_confidence[0] << std::endl;
    }
  }

  std::cout << "Saved debug images to: " << out_dir << std::endl;
  return 0;
}

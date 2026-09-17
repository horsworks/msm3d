#include "msm3d/msm_calibration.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

int main(int argc, char** argv) {
  const std::string phase_path =
      (argc > 1) ? argv[1] : "./output/phase/pose_5.exr";
  const std::string out_dir = "./output/debug";

  if (!std::filesystem::exists(phase_path)) {
    std::cerr << "Phase map not found: " << phase_path << std::endl;
    return -1;
  }

  cv::Mat phase_raw = cv::imread(phase_path, cv::IMREAD_UNCHANGED);
  if (phase_raw.empty()) {
    std::cerr << "Failed to load phase map: " << phase_path << std::endl;
    return -1;
  }

  // 提取单通道浮点
  cv::Mat phase_f64;
  if (phase_raw.channels() > 1) {
    cv::extractChannel(phase_raw, phase_f64, 0);
  } else {
    phase_f64 = phase_raw;
  }
  phase_f64.convertTo(phase_f64, CV_64F);

  double min_val, max_val;
  cv::minMaxLoc(phase_f64, &min_val, &max_val);
  std::cout << "Phase map loaded: " << phase_path << std::endl;
  std::cout << "Image size: " << phase_f64.cols << " x " << phase_f64.rows
            << std::endl;
  std::cout << "Phase range: [" << min_val << ", " << max_val << "]"
            << std::endl;

  // 生成伪彩色底图便于人眼观察
  cv::Mat norm_phase, gray_8u, color_vis;
  cv::normalize(phase_f64, norm_phase, 0, 255, cv::NORM_MINMAX);
  norm_phase.convertTo(gray_8u, CV_8U);
  cv::applyColorMap(gray_8u, color_vis, cv::COLORMAP_VIRIDIS);

  // 挑选 3 个代表性相位点 (低、中、高)
  const std::vector<double> test_psis = {35.0, 90.0, 145.0};
  const std::vector<cv::Scalar> colors = {
      cv::Scalar(0, 0, 255),     // Red
      cv::Scalar(0, 255, 0),     // Green
      cv::Scalar(255, 255, 255)  // White
  };

  std::cout << "\n--- Subpixel Extraction Diagnostic ---" << std::endl;

  for (std::size_t i = 0; i < test_psis.size(); ++i) {
    const double target_psi = test_psis[i];
    const auto subpixels =
        msm3d::extractIsoPhaseSubpixels(phase_f64, target_psi);

    // 统计每行的提取频次
    std::vector<int> row_hits(phase_f64.rows, 0);
    for (const auto& pt : subpixels) {
      const int y = static_cast<int>(std::round(pt.y));
      if (y >= 0 && y < phase_f64.rows) {
        row_hits[y]++;
      }
    }

    int active_rows = 0;
    int max_hits_per_row = 0;
    int multi_hit_rows = 0;
    for (int y = 0; y < phase_f64.rows; ++y) {
      if (row_hits[y] > 0) {
        active_rows++;
        if (row_hits[y] > 1) {
          multi_hit_rows++;
        }
        max_hits_per_row = std::max(max_hits_per_row, row_hits[y]);
      }
    }

    std::cout << "Target Psi: " << std::fixed << std::setprecision(1)
              << target_psi << " | Extracted Points: " << subpixels.size()
              << " | Active Rows: " << active_rows << " / " << phase_f64.rows
              << " | Rows with Multi-Crossings: " << multi_hit_rows
              << " | Max Points on Single Row: " << max_hits_per_row
              << std::endl;

    // 在底图上绘制亚像素点
    for (const auto& pt : subpixels) {
      cv::circle(color_vis, pt, 1, colors[i], -1);
    }
  }

  std::filesystem::create_directories(out_dir);
  const std::string out_img_path = out_dir + "/inspect_iso_phase.png";
  cv::imwrite(out_img_path, color_vis);
  std::cout << "\nVisual diagnostic map saved to: " << out_img_path
            << std::endl;

  return 0;
}
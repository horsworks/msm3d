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

  const int cols = phase_f64.cols;
  const int rows = phase_f64.rows;

  std::cout << "Image size: " << cols << " x " << rows << std::endl;

  // 1. 终端网格数值采样输出 (采样 5 行 x 6 列)
  std::cout << "\n--- Spatial Phase Grid Samples (rad) ---" << std::endl;
  std::cout << std::setw(8) << "y \\ x";
  for (int x = 200; x < cols; x += 350) {
    std::cout << std::setw(12) << ("x=" + std::to_string(x));
  }
  std::cout << std::endl;

  for (int y = 200; y < rows; y += 250) {
    std::cout << std::setw(8) << ("y=" + std::to_string(y));
    const double* r_ptr = phase_f64.ptr<double>(y);
    for (int x = 200; x < cols; x += 350) {
      std::cout << std::setw(12) << std::fixed << std::setprecision(1)
                << r_ptr[x];
    }
    std::cout << std::endl;
  }

  // 2. 生成中心扫描行 (y = rows / 2) 的截面曲线图
  const int mid_y = rows / 2;
  const double* mid_row = phase_f64.ptr<double>(mid_y);

  const int plot_w = 1000;
  const int plot_h = 500;
  cv::Mat plot_img = cv::Mat::zeros(plot_h, plot_w, CV_8UC3);
  plot_img.setTo(cv::Scalar(245, 245, 245));

  // 确定有效纵坐标刻度范围 [0, 480]
  const double max_plot_psi = 480.0;
  auto to_screen = [&](int img_x, double psi) -> cv::Point {
    const int sx = static_cast<int>(
        static_cast<double>(img_x) / cols * (plot_w - 100) + 60);
    const double clamped_psi = std::max(0.0, std::min(psi, max_plot_psi));
    const int sy = static_cast<int>(
        (plot_h - 60) - (clamped_psi / max_plot_psi) * (plot_h - 100));
    return cv::Point(sx, sy);
  };

  // 绘制坐标轴与网格线
  cv::line(plot_img, cv::Point(60, plot_h - 60),
           cv::Point(plot_w - 40, plot_h - 60), cv::Scalar(0, 0, 0), 2);
  cv::line(plot_img, cv::Point(60, 40), cv::Point(60, plot_h - 60),
           cv::Scalar(0, 0, 0), 2);
  cv::putText(plot_img, "Phase (rad)", cv::Point(15, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  cv::putText(plot_img, "Image Column X (px)",
              cv::Point(plot_w / 2 - 50, plot_h - 20), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(0, 0, 0), 1);

  for (double p = 0; p <= 450; p += 50) {
    const cv::Point pt = to_screen(0, p);
    cv::line(plot_img, cv::Point(55, pt.y), cv::Point(plot_w - 40, pt.y),
             cv::Scalar(210, 210, 210), 1);
    cv::putText(plot_img, std::to_string(static_cast<int>(p)),
                cv::Point(20, pt.y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(80, 80, 80), 1);
  }

  // 绘制中行相位变化曲线
  for (int x = 1; x < cols; ++x) {
    const cv::Point p_prev = to_screen(x - 1, mid_row[x - 1]);
    const cv::Point p_curr = to_screen(x, mid_row[x]);
    cv::line(plot_img, p_prev, p_curr, cv::Scalar(200, 50, 0), 2);
  }

  // 3. 生成全视场等相位线概览图 (从 30 到 150，步长 30)
  cv::Mat norm_phase, gray_8u, vis_contours;
  cv::normalize(phase_f64, norm_phase, 0, 255, cv::NORM_MINMAX);
  norm_phase.convertTo(gray_8u, CV_8U);
  cv::applyColorMap(gray_8u, vis_contours, cv::COLORMAP_VIRIDIS);

  std::cout << "\n--- Extracting Full-Field Iso-Phase Contours ---"
            << std::endl;
  for (double target_psi = 30.0; target_psi <= 150.0; target_psi += 30.0) {
    const auto subpixels =
        msm3d::extractIsoPhaseSubpixels(phase_f64, target_psi);
    if (subpixels.empty()) {
      continue;
    }

    std::cout << "Contour psi=" << std::setw(5) << target_psi
              << " | points=" << subpixels.size() << " | x_mean="
              << static_cast<int>(subpixels[subpixels.size() / 2].x)
              << std::endl;

    for (std::size_t k = 0; k < subpixels.size(); ++k) {
      cv::circle(vis_contours, subpixels[k], 2, cv::Scalar(255, 255, 255), -1);
      if (k > 0 && std::abs(subpixels[k].y - subpixels[k - 1].y) <= 3.0) {
        cv::line(vis_contours, subpixels[k - 1], subpixels[k],
                 cv::Scalar(255, 255, 255), 2);
      }
    }

    // 标注文字
    const auto& mid_pt = subpixels[subpixels.size() / 2];
    cv::putText(
        vis_contours, std::to_string(static_cast<int>(target_psi)),
        cv::Point(static_cast<int>(mid_pt.x) + 5, static_cast<int>(mid_pt.y)),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
  }

  std::filesystem::create_directories(out_dir);
  const std::string curve_path = out_dir + "/phase_profile_curve.png";
  const std::string contour_path = out_dir + "/phase_contours_overview.png";

  cv::imwrite(curve_path, plot_img);
  cv::imwrite(contour_path, vis_contours);

  std::cout << "\nSaved profile curve to: " << curve_path << std::endl;
  std::cout << "Saved contour overview to: " << contour_path << std::endl;

  return 0;
}
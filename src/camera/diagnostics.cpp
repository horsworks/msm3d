#include "msm3d/camera/diagnostics.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace msm3d {
namespace {

struct CameraPointResidual {
  std::size_t view_index = 0;
  std::size_t point_index = 0;
  int board_row = 0;
  int board_col = 0;
  cv::Point3f object_point;
  cv::Point2f observed_point;
  cv::Point2f projected_point;
  cv::Point2d residual;
  double error = 0.0;
};

struct CameraResidualStatistics {
  std::size_t count = 0;
  double mean_dx = 0.0;
  double mean_dy = 0.0;
  double mean_error = 0.0;
  double rms_error = 0.0;
  double p95_error = 0.0;
  double max_error = 0.0;
};

double percentile(std::vector<double> values, double q) {
  if (values.empty()) {
    return 0.0;
  }

  q = std::clamp(q, 0.0, 1.0);
  const double position = q * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));

  std::nth_element(values.begin(), values.begin() + lower, values.end());
  const double lower_value = values[lower];

  if (upper == lower) {
    return lower_value;
  }

  std::nth_element(values.begin(), values.begin() + upper, values.end());
  const double upper_value = values[upper];
  const double t = position - static_cast<double>(lower);
  return lower_value + t * (upper_value - lower_value);
}

CameraResidualStatistics summarizeResiduals(
    const std::vector<const CameraPointResidual*>& residuals) {
  CameraResidualStatistics stats;
  if (residuals.empty()) {
    return stats;
  }

  double sum_dx = 0.0;
  double sum_dy = 0.0;
  double sum_error = 0.0;
  double sum_squared_error = 0.0;
  std::vector<double> errors;
  errors.reserve(residuals.size());

  for (const CameraPointResidual* residual : residuals) {
    if (residual == nullptr) {
      continue;
    }

    sum_dx += residual->residual.x;
    sum_dy += residual->residual.y;
    sum_error += residual->error;
    sum_squared_error += residual->error * residual->error;
    errors.push_back(residual->error);
    stats.max_error = std::max(stats.max_error, residual->error);
  }

  stats.count = errors.size();
  if (stats.count == 0) {
    return stats;
  }

  const double count = static_cast<double>(stats.count);
  stats.mean_dx = sum_dx / count;
  stats.mean_dy = sum_dy / count;
  stats.mean_error = sum_error / count;
  stats.rms_error = std::sqrt(sum_squared_error / count);
  stats.p95_error = percentile(errors, 0.95);
  return stats;
}

CameraResidualStatistics summarizeResiduals(
    const std::vector<CameraPointResidual>& residuals) {
  std::vector<const CameraPointResidual*> pointers;
  pointers.reserve(residuals.size());
  for (const auto& residual : residuals) {
    pointers.push_back(&residual);
  }
  return summarizeResiduals(pointers);
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }

  std::string escaped = "\"";
  for (const char ch : value) {
    if (ch == '\"') {
      escaped += "\"\"";
    } else {
      escaped += ch;
    }
  }
  escaped += "\"";
  return escaped;
}

std::vector<CameraPointResidual> buildCameraPointResiduals(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const CameraCalibrationResult& result,
    const CalibrationBoard& board) {
  if (object_points.size() != image_points.size() ||
      object_points.size() != result.rotation_vectors.size() ||
      object_points.size() != result.translation_vectors.size()) {
    throw std::runtime_error(
        "Cannot build residual diagnostics: inconsistent calibration view "
        "counts.");
  }

  std::vector<CameraPointResidual> residuals;
  std::size_t total_points = 0;
  for (const auto& points : image_points) {
    total_points += points.size();
  }
  residuals.reserve(total_points);

  for (std::size_t view = 0; view < object_points.size(); ++view) {
    if (object_points[view].size() != image_points[view].size()) {
      throw std::runtime_error(
          "Cannot build residual diagnostics: object/image point count "
          "mismatch.");
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points[view], result.rotation_vectors[view],
                      result.translation_vectors[view], result.camera_matrix,
                      result.distortion_coefficients, projected_points);

    if (projected_points.size() != image_points[view].size()) {
      throw std::runtime_error(
          "Cannot build residual diagnostics: projected point count "
          "mismatch.");
    }

    for (std::size_t point = 0; point < image_points[view].size(); ++point) {
      CameraPointResidual residual;
      residual.view_index = view;
      residual.point_index = point;
      residual.board_row =
          board.columns > 0 ? static_cast<int>(point) / board.columns : 0;
      residual.board_col =
          board.columns > 0 ? static_cast<int>(point) % board.columns : 0;
      residual.object_point = object_points[view][point];
      residual.observed_point = image_points[view][point];
      residual.projected_point = projected_points[point];
      residual.residual =
          cv::Point2d(static_cast<double>(residual.observed_point.x -
                                          residual.projected_point.x),
                      static_cast<double>(residual.observed_point.y -
                                          residual.projected_point.y));
      residual.error = std::hypot(residual.residual.x, residual.residual.y);
      residuals.push_back(residual);
    }
  }

  return residuals;
}

void printWorstCameraViewsImpl(const CameraCalibrationResult& result,
                     const std::vector<std::string>& valid_view_names,
                     std::size_t max_count) {
  if (result.per_view_errors.empty()) {
    return;
  }

  std::vector<std::size_t> order(result.per_view_errors.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&result](std::size_t lhs, std::size_t rhs) {
              return result.per_view_errors[lhs] > result.per_view_errors[rhs];
            });

  const std::size_t count = std::min(max_count, order.size());
  std::cout << "\nWorst views by reprojection RMS:" << std::endl;
  for (std::size_t rank = 0; rank < count; ++rank) {
    const std::size_t i = order[rank];
    const std::string name = i < valid_view_names.size()
                                 ? valid_view_names[i]
                                 : ("view_" + std::to_string(i));

    std::cout << "  " << std::setw(2) << rank + 1 << ". " << name
              << ": mean=" << result.per_view_mean_errors[i]
              << " px, rms=" << result.per_view_errors[i]
              << " px, p95=" << result.per_view_p95_errors[i]
              << " px, max=" << result.per_view_max_errors[i] << " px"
              << std::endl;
  }
}

void printCameraIntrinsicStdDeviationsImpl(const cv::Mat& stddev) {
  if (stddev.empty() || stddev.total() < 4) {
    return;
  }

  cv::Mat flat = stddev.reshape(1, 1);
  cv::Mat flat64;
  flat.convertTo(flat64, CV_64F);

  std::cout << "Intrinsic standard deviations:" << std::endl;
  std::cout << "  fx=" << flat64.at<double>(0, 0)
            << ", fy=" << flat64.at<double>(0, 1)
            << ", cx=" << flat64.at<double>(0, 2)
            << ", cy=" << flat64.at<double>(0, 3) << std::endl;

  static const char* names[] = {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"};
  const int distortion_count =
      std::min<int>(8, static_cast<int>(flat64.total()) - 4);
  if (distortion_count > 0) {
    std::cout << "  distortion:";
    for (int i = 0; i < distortion_count; ++i) {
      std::cout << " " << names[i] << "=" << flat64.at<double>(0, 4 + i);
    }
    std::cout << std::endl;
  }
}

void printWorstBoardPoints(const std::vector<CameraPointResidual>& residuals,
                           const CalibrationBoard& board,
                           std::size_t max_count) {
  const int point_count = board.columns * board.rows;
  if (point_count <= 0) {
    return;
  }

  struct GridStats {
    int point_index = 0;
    int row = 0;
    int col = 0;
    CameraResidualStatistics stats;
  };

  std::vector<std::vector<const CameraPointResidual*>> grouped(
      static_cast<std::size_t>(point_count));
  for (const auto& residual : residuals) {
    if (residual.point_index < grouped.size()) {
      grouped[residual.point_index].push_back(&residual);
    }
  }

  std::vector<GridStats> stats;
  stats.reserve(grouped.size());
  for (int index = 0; index < point_count; ++index) {
    GridStats entry;
    entry.point_index = index;
    entry.row = index / board.columns;
    entry.col = index % board.columns;
    entry.stats = summarizeResiduals(grouped[static_cast<std::size_t>(index)]);
    stats.push_back(entry);
  }

  std::sort(stats.begin(), stats.end(),
            [](const GridStats& lhs, const GridStats& rhs) {
              return lhs.stats.rms_error > rhs.stats.rms_error;
            });

  std::cout << "\nWorst board grid points by cross-view RMS:" << std::endl;
  const std::size_t count = std::min(max_count, stats.size());
  for (std::size_t rank = 0; rank < count; ++rank) {
    const auto& entry = stats[rank];
    std::cout << "  " << std::setw(2) << rank + 1 << ". (row=" << entry.row
              << ", col=" << entry.col << ")"
              << ": mean_vec=(" << entry.stats.mean_dx << ", "
              << entry.stats.mean_dy << ") px"
              << ", mean=" << entry.stats.mean_error << " px"
              << ", rms=" << entry.stats.rms_error << " px"
              << ", p95=" << entry.stats.p95_error << " px"
              << ", max=" << entry.stats.max_error << " px"
              << ", views=" << entry.stats.count << std::endl;
  }
}

void printWorstObservations(const std::vector<CameraPointResidual>& residuals,
                            const std::vector<std::string>& valid_view_names,
                            std::size_t max_count) {
  std::vector<const CameraPointResidual*> order;
  order.reserve(residuals.size());
  for (const auto& residual : residuals) {
    order.push_back(&residual);
  }

  std::sort(order.begin(), order.end(),
            [](const CameraPointResidual* lhs, const CameraPointResidual* rhs) {
              return lhs->error > rhs->error;
            });

  std::cout << "\nWorst individual reprojection observations:" << std::endl;
  const std::size_t count = std::min(max_count, order.size());
  for (std::size_t rank = 0; rank < count; ++rank) {
    const CameraPointResidual& residual = *order[rank];
    const std::string name =
        residual.view_index < valid_view_names.size()
            ? valid_view_names[residual.view_index]
            : ("view_" + std::to_string(residual.view_index));

    std::cout << "  " << std::setw(2) << rank + 1 << ". " << name
              << " (row=" << residual.board_row
              << ", col=" << residual.board_col << ")"
              << ": error=" << residual.error << " px"
              << ", residual=(" << residual.residual.x << ", "
              << residual.residual.y << ") px"
              << ", image=(" << residual.observed_point.x << ", "
              << residual.observed_point.y << ") px" << std::endl;
  }
}

bool writeObservationCsv(
    const std::filesystem::path& path,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::string>& valid_view_names,
    const CalibrationBoard& board) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  out << "view_index,filename,point_index,board_row,board_col,"
         "object_x_mm,object_y_mm,object_z_mm,image_x_px,image_y_px\n";
  out << std::setprecision(12);

  for (std::size_t view = 0; view < image_points.size(); ++view) {
    const std::string name = view < valid_view_names.size()
                                 ? valid_view_names[view]
                                 : ("view_" + std::to_string(view));

    for (std::size_t point = 0; point < image_points[view].size(); ++point) {
      const int row =
          board.columns > 0 ? static_cast<int>(point) / board.columns : 0;
      const int col =
          board.columns > 0 ? static_cast<int>(point) % board.columns : 0;
      const cv::Point3f object = object_points[view][point];
      const cv::Point2f image = image_points[view][point];

      out << view << ',' << csvEscape(name) << ',' << point << ',' << row << ','
          << col << ',' << object.x << ',' << object.y << ',' << object.z << ','
          << image.x << ',' << image.y << '\n';
    }
  }

  return true;
}

bool writeResidualCsv(const std::filesystem::path& path,
                      const std::vector<CameraPointResidual>& residuals,
                      const std::vector<std::string>& valid_view_names) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  out << "view_index,filename,point_index,board_row,board_col,"
         "object_x_mm,object_y_mm,object_z_mm,"
         "observed_x_px,observed_y_px,projected_x_px,projected_y_px,"
         "residual_x_px,residual_y_px,error_px\n";
  out << std::setprecision(12);

  for (const auto& residual : residuals) {
    const std::string name =
        residual.view_index < valid_view_names.size()
            ? valid_view_names[residual.view_index]
            : ("view_" + std::to_string(residual.view_index));

    out << residual.view_index << ',' << csvEscape(name) << ','
        << residual.point_index << ',' << residual.board_row << ','
        << residual.board_col << ',' << residual.object_point.x << ','
        << residual.object_point.y << ',' << residual.object_point.z << ','
        << residual.observed_point.x << ',' << residual.observed_point.y << ','
        << residual.projected_point.x << ',' << residual.projected_point.y
        << ',' << residual.residual.x << ',' << residual.residual.y << ','
        << residual.error << '\n';
  }

  return true;
}

bool writeGridResidualSummaryCsv(const std::filesystem::path& path,
                                 const std::vector<CameraPointResidual>& residuals,
                                 const CalibrationBoard& board) {
  const int point_count = board.columns * board.rows;
  if (point_count <= 0) {
    return false;
  }

  std::vector<std::vector<const CameraPointResidual*>> grouped(
      static_cast<std::size_t>(point_count));
  for (const auto& residual : residuals) {
    if (residual.point_index < grouped.size()) {
      grouped[residual.point_index].push_back(&residual);
    }
  }

  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  out << "point_index,board_row,board_col,count,mean_residual_x_px,"
         "mean_residual_y_px,mean_error_px,rms_error_px,p95_error_px,"
         "max_error_px\n";
  out << std::setprecision(12);

  for (int index = 0; index < point_count; ++index) {
    const CameraResidualStatistics stats =
        summarizeResiduals(grouped[static_cast<std::size_t>(index)]);
    out << index << ',' << index / board.columns << ',' << index % board.columns
        << ',' << stats.count << ',' << stats.mean_dx << ',' << stats.mean_dy
        << ',' << stats.mean_error << ',' << stats.rms_error << ','
        << stats.p95_error << ',' << stats.max_error << '\n';
  }

  return true;
}

bool writeImageResidualBinsCsv(const std::filesystem::path& path,
                               const std::vector<CameraPointResidual>& residuals,
                               const cv::Size& image_size, int bins_x,
                               int bins_y) {
  if (image_size.width <= 0 || image_size.height <= 0 || bins_x <= 0 ||
      bins_y <= 0) {
    return false;
  }

  const std::size_t bin_count = static_cast<std::size_t>(bins_x * bins_y);
  std::vector<std::vector<const CameraPointResidual*>> grouped(bin_count);

  for (const auto& residual : residuals) {
    const double normalized_x = static_cast<double>(residual.observed_point.x) /
                                static_cast<double>(image_size.width);
    const double normalized_y = static_cast<double>(residual.observed_point.y) /
                                static_cast<double>(image_size.height);

    const int bin_x = std::clamp(
        static_cast<int>(std::floor(normalized_x * bins_x)), 0, bins_x - 1);
    const int bin_y = std::clamp(
        static_cast<int>(std::floor(normalized_y * bins_y)), 0, bins_y - 1);

    const std::size_t index = static_cast<std::size_t>(bin_y * bins_x + bin_x);
    grouped[index].push_back(&residual);
  }

  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }

  out << "bin_x,bin_y,x_min_px,x_max_px,y_min_px,y_max_px,count,"
         "mean_residual_x_px,mean_residual_y_px,mean_error_px,"
         "rms_error_px,p95_error_px,max_error_px\n";
  out << std::setprecision(12);

  for (int bin_y = 0; bin_y < bins_y; ++bin_y) {
    for (int bin_x = 0; bin_x < bins_x; ++bin_x) {
      const std::size_t index =
          static_cast<std::size_t>(bin_y * bins_x + bin_x);
      const CameraResidualStatistics stats = summarizeResiduals(grouped[index]);

      const double x_min =
          static_cast<double>(bin_x) * image_size.width / bins_x;
      const double x_max =
          static_cast<double>(bin_x + 1) * image_size.width / bins_x;
      const double y_min =
          static_cast<double>(bin_y) * image_size.height / bins_y;
      const double y_max =
          static_cast<double>(bin_y + 1) * image_size.height / bins_y;

      out << bin_x << ',' << bin_y << ',' << x_min << ',' << x_max << ','
          << y_min << ',' << y_max << ',' << stats.count << ',' << stats.mean_dx
          << ',' << stats.mean_dy << ',' << stats.mean_error << ','
          << stats.rms_error << ',' << stats.p95_error << ',' << stats.max_error
          << '\n';
    }
  }

  return true;
}

void saveCameraResidualDiagnosticsImpl(
    const std::filesystem::path& diagnostics_dir,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::string>& valid_view_names,
    const CameraCalibrationResult& result,
    const CalibrationBoard& board, const cv::Size& image_size) {
  std::filesystem::create_directories(diagnostics_dir);

  const std::vector<CameraPointResidual> residuals =
      buildCameraPointResiduals(object_points, image_points, result, board);
  const CameraResidualStatistics global_stats = summarizeResiduals(residuals);

  std::cout << "\nResidual field diagnostics:" << std::endl;
  std::cout << "  global mean residual vector: (" << global_stats.mean_dx
            << ", " << global_stats.mean_dy << ") px, magnitude="
            << std::hypot(global_stats.mean_dx, global_stats.mean_dy) << " px"
            << std::endl;

  printWorstBoardPoints(residuals, board, 10);
  printWorstObservations(residuals, valid_view_names, 10);

  const auto observations_path = diagnostics_dir / "camera_observations.csv";
  const auto residuals_path =
      diagnostics_dir / "camera_reprojection_residuals.csv";
  const auto grid_path = diagnostics_dir / "camera_grid_residual_summary.csv";
  const auto image_bins_path =
      diagnostics_dir / "camera_image_residual_bins.csv";

  const bool observations_ok = writeObservationCsv(
      observations_path, object_points, image_points, valid_view_names, board);
  const bool residuals_ok =
      writeResidualCsv(residuals_path, residuals, valid_view_names);
  const bool grid_ok = writeGridResidualSummaryCsv(grid_path, residuals, board);
  const bool image_bins_ok =
      writeImageResidualBinsCsv(image_bins_path, residuals, image_size, 8, 5);

  std::cout << "\nResidual diagnostics files:" << std::endl;
  std::cout << "  " << observations_path.string()
            << (observations_ok ? "" : " [FAILED]") << std::endl;
  std::cout << "  " << residuals_path.string()
            << (residuals_ok ? "" : " [FAILED]") << std::endl;
  std::cout << "  " << grid_path.string() << (grid_ok ? "" : " [FAILED]")
            << std::endl;
  std::cout << "  " << image_bins_path.string()
            << (image_bins_ok ? "" : " [FAILED]") << std::endl;
}


}  // namespace

void printCameraIntrinsicStdDeviations(const cv::Mat& stddev) {
  printCameraIntrinsicStdDeviationsImpl(stddev);
}

void printWorstCameraViews(const CameraCalibrationResult& result,
                           const std::vector<std::string>& valid_view_names,
                           std::size_t max_count) {
  printWorstCameraViewsImpl(result, valid_view_names, max_count);
}

void saveCameraResidualDiagnostics(
    const std::filesystem::path& diagnostics_dir,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::string>& valid_view_names,
    const CameraCalibrationResult& result, const CalibrationBoard& board,
    const cv::Size& image_size) {
  saveCameraResidualDiagnosticsImpl(diagnostics_dir, object_points, image_points,
                                    valid_view_names, result, board, image_size);
}

}  // namespace msm3d

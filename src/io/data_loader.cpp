#include "msm3d/io/data_loader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace msm3d {
namespace {

// 内部自然排序比较函数，支持 img_1, img_2, ..., img_10 正确排序
bool naturalCompare(const std::string& a, const std::string& b) {
  std::size_t i = 0, j = 0;
  const std::size_t len_a = a.size(), len_b = b.size();

  while (i < len_a && j < len_b) {
    if (std::isdigit(static_cast<unsigned char>(a[i])) &&
        std::isdigit(static_cast<unsigned char>(b[j]))) {
      std::size_t zeros_a = 0, zeros_b = 0;
      while (i < len_a && a[i] == '0') {
        ++zeros_a;
        ++i;
      }
      while (j < len_b && b[j] == '0') {
        ++zeros_b;
        ++j;
      }

      std::size_t start_a = i, start_b = j;
      while (i < len_a && std::isdigit(static_cast<unsigned char>(a[i]))) ++i;
      while (j < len_b && std::isdigit(static_cast<unsigned char>(b[j]))) ++j;

      std::size_t num_len_a = i - start_a;
      std::size_t num_len_b = j - start_b;

      if (num_len_a != num_len_b) {
        return num_len_a < num_len_b;
      }

      while (start_a < i && start_b < j) {
        if (a[start_a] != b[start_b]) {
          return a[start_a] < b[start_b];
        }
        ++start_a;
        ++start_b;
      }

      if (zeros_a != zeros_b) {
        return zeros_a > zeros_b;
      }
    } else {
      if (a[i] != b[j]) {
        return a[i] < b[j];
      }
      ++i;
      ++j;
    }
  }

  return len_a < len_b;
}

}  // namespace

std::vector<std::string> DataLoader::scanDirectory(
    const std::string& folder, const std::string& extension) {
  std::filesystem::path folder_path(folder);
  if (!std::filesystem::exists(folder_path)) {
    throw std::runtime_error("Directory does not exist: " + folder);
  }

  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      if (extension.empty() || entry.path().extension().string() == extension) {
        files.push_back(entry.path().string());
      }
    }
  }

  std::sort(files.begin(), files.end(), naturalCompare);
  return files;
}

std::vector<std::string> DataLoader::loadCameraCalibImagePaths(
    const std::string& folder) {
  std::filesystem::path folder_path(folder);
  if (!std::filesystem::exists(folder_path)) {
    throw std::runtime_error("Calibration directory does not exist: " + folder);
  }

  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      const std::string ext = entry.path().extension().string();
      if (ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".tif") {
        files.push_back(entry.path().string());
      }
    }
  }

  std::sort(files.begin(), files.end(), naturalCompare);
  return files;
}

std::vector<std::vector<std::string>> DataLoader::loadPoseFringePaths(
    const std::vector<std::string>& all_sorted_files, int pose_id,
    const FringePatternConfig& pattern) {
  const std::size_t num_freqs = pattern.frequencies.size();
  const std::size_t images_per_pose =
      static_cast<std::size_t>(pattern.steps) * num_freqs;
  const std::size_t start_idx =
      static_cast<std::size_t>(pose_id) * images_per_pose;
  const std::size_t end_idx = start_idx + images_per_pose;

  if (all_sorted_files.size() < end_idx) {
    throw std::runtime_error(
        "Insufficient fringe files for pose_id " + std::to_string(pose_id) +
        ". Required index up to: " + std::to_string(end_idx) +
        ", but total files count is: " +
        std::to_string(all_sorted_files.size()));
  }

  std::vector<std::vector<std::string>> pose_paths(num_freqs);
  for (std::size_t f = 0; f < num_freqs; ++f) {
    pose_paths[f].reserve(pattern.steps);
    for (int s = 0; s < pattern.steps; ++s) {
      pose_paths[f].push_back(
          all_sorted_files[start_idx + f * pattern.steps + s]);
    }
  }

  return pose_paths;
}

std::vector<cv::Mat> DataLoader::loadImages(
    const std::vector<std::string>& paths, int flags) {
  std::vector<cv::Mat> images;
  images.reserve(paths.size());

  for (const auto& path : paths) {
    cv::Mat img = cv::imread(path, flags);
    if (img.empty()) {
      throw std::runtime_error("Failed to read image or file is corrupted: " +
                               path);
    }
    images.push_back(std::move(img));
  }

  return images;
}

}  // namespace msm3d
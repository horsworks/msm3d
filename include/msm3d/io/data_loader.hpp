#pragma once

#include "msm3d/io/config.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

class DataLoader {
 public:
  static std::vector<std::string> scanDirectory(
      const std::string& folder, const std::string& extension = "");

  static std::vector<std::string> loadCameraCalibImagePaths(
      const std::string& folder);

  static std::vector<std::vector<std::string>> loadPoseFringePaths(
      const std::vector<std::string>& all_sorted_files, int pose_id,
      const FringePatternConfig& pattern);

  static std::vector<cv::Mat> loadImages(const std::vector<std::string>& paths,
                                         int flags = 0);
};

}  // namespace msm3d

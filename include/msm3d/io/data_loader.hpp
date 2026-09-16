#pragma once

#include "msm3d/io/config.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace msm3d {

class DataLoader {
 public:
  // 1. 扫描指定目录下的指定后缀文件，并按自然排序（Natural
  // Sort）返回绝对路径列表
  static std::vector<std::string> scanDirectory(
      const std::string& folder, const std::string& extension = "");

  // 2. 标定图像加载接口（兼容 .bmp, .png, .jpg, .tif 等常用格式）
  static std::vector<std::string> loadCameraCalibImagePaths(
      const std::string& folder);

  // 3. 多位姿条纹图路径分组索引
  //    返回外层索引为频率（0 ~ frequencies.size()-1），内层索引为相移步数（0 ~
  //    steps-1）的二维路径表
  static std::vector<std::vector<std::string>> loadPoseFringePaths(
      const std::vector<std::string>& all_sorted_files, int pose_id,
      const FringePatternConfig& pattern);

  // 4. 根据给定的文件路径列表批量读取为灰度 Mat
  static std::vector<cv::Mat> loadImages(
      const std::vector<std::string>& paths,
      int flags = 0 /* cv::IMREAD_GRAYSCALE */);
};

}  // namespace msm3d
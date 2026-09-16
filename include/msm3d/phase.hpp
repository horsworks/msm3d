#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

struct PhaseConfig {
  std::string fringe_folder;
  std::string extension;
  int pose_count = 0;
  int steps = 24;
  std::vector<int> frequencies;
  std::string output_folder;
};

struct PosePhaseImages {
  int pose_id = -1;
  // images[freq_index][step_index]
  std::vector<std::vector<std::string>> images;
};

// 与相机标定模块风格对齐的独立配置加载函数
PhaseConfig loadPhaseConfig(const std::string& yaml_file);

class PhaseProcessor {
 public:
  // 静态或无状态接口，传入配置以解耦数据与算法
  PosePhaseImages loadPoseImages(int pose_id, const PhaseConfig& config) const;

  cv::Mat computeWrappedPhase(const std::vector<cv::Mat>& images) const;

  cv::Mat computeAbsolutePhase(const std::vector<cv::Mat>& wrapped,
                               const std::vector<int>& frequencies) const;

  bool savePhaseEXR(const cv::Mat& phase, const std::string& filename) const;
};

}  // namespace msm3d

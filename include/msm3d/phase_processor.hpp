#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace msm3d {

class PhaseProcessor {
 public:
  // N 步相移：求解包裹相位
  static cv::Mat computeWrappedPhase(const std::vector<cv::Mat>& images,
                                     cv::Mat* out_modulation = nullptr);

  // 外差解包裹 (保持兼容)
  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies);

  // 增强版外差解包裹：结合多频调制度进行信噪比截断 (暗区/黑圆点直接置为 NaN)
  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies,
      const std::vector<cv::Mat>& modulations, double min_modulation = 8.0);

  // 掩膜感知中值滤波 (消除激光微观毛刺，不污染 NaN 边界)
  static cv::Mat filterPhaseNoise(const cv::Mat& phase, int kernel_size = 3);

  // 5. 保存 EXR 绝对相位图 (保持原签名)
  static bool savePhaseEXR(const cv::Mat& phase, const std::string& filename,
                           bool compress = false);
};

}  // namespace msm3d
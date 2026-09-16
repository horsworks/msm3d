#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace msm3d {

class PhaseProcessor {
 public:
  // N 步相移计算包裹相位（输出范围 [0, 2*pi) 的 CV_64F 矩阵）
  // 若 out_modulation !=
  // nullptr，则额外计算并写回调制度图，否则跳过开方与额外内存分配
  static cv::Mat computeWrappedPhase(const std::vector<cv::Mat>& images,
                                     cv::Mat* out_modulation = nullptr);

  // 多频外差绝对相位解算（输入 3 组包裹相位矩阵与 3 个降序频率值）
  static cv::Mat computeAbsolutePhase(
      const std::vector<cv::Mat>& wrapped_phases,
      const std::vector<int>& frequencies);

  // 保存高动态范围浮点相位为 EXR 格式（支持开启/关闭压缩）
  static bool savePhaseEXR(const cv::Mat& phase, const std::string& filename,
                           bool compress = false);
};

}  // namespace msm3d
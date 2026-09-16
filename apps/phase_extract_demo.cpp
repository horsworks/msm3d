#include "msm3d/phase.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>

int main() {
  msm3d::PhaseConfig config;
  try {
    config = msm3d::loadPhaseConfig("./config/phase.yaml");
  } catch (const std::exception& e) {
    std::cerr << "Load config failed: " << e.what() << std::endl;
    return -1;
  }

  // 优先使用配置文件中指定的输出路径，若为空则回退到默认路径
  const std::string output_dir =
      config.output_folder.empty() ? "./output/phase" : config.output_folder;
  std::filesystem::create_directories(output_dir);

  msm3d::PhaseProcessor processor;

  // 使用配置中的实际位姿数量替代硬编码的 27
  for (int pose = 0; pose < config.pose_count; ++pose) {
    try {
      // 传入 config 供文件定位使用
      auto data = processor.loadPoseImages(pose, config);

      std::vector<cv::Mat> wrapped;
      wrapped.reserve(data.images.size());

      for (const auto& freq_files : data.images) {
        std::vector<cv::Mat> imgs;
        imgs.reserve(freq_files.size());

        for (const auto& file : freq_files) {
          cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
          if (img.empty()) {
            std::cerr << "Warning: Failed to load image: " << file << std::endl;
          }
          imgs.push_back(img);
        }

        wrapped.push_back(processor.computeWrappedPhase(imgs));
      }

      // 直接使用 config.frequencies，避免硬编码 {5, 3, 1}
      cv::Mat phase =
          processor.computeAbsolutePhase(wrapped, config.frequencies);

      const std::string save_path =
          output_dir + "/pose_" + std::to_string(pose) + ".exr";
      if (!processor.savePhaseEXR(phase, save_path)) {
        std::cerr << "Failed to save EXR for pose " << pose << std::endl;
      }

      std::cout << "Pose " << pose << " done" << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "Error processing pose " << pose << ": " << e.what()
                << std::endl;
    }
  }

  return 0;
}
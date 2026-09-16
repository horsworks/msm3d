#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"
#include "msm3d/phase_processor.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/msm_calibration.yaml";

  // 1. 加载配置
  msm3d::PhaseConfig config;
  try {
    config = msm3d::loadPhaseConfig(config_path);
    std::cout << "Loaded phase configuration from: " << config_path
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load configuration: " << e.what() << std::endl;
    return -1;
  }

  const std::string output_dir =
      config.output_folder.empty() ? "./output/phase" : config.output_folder;
  std::filesystem::create_directories(output_dir);

  // 2. 全局单次扫描条纹图片并完成自然排序
  std::vector<std::string> all_files;
  try {
    all_files = msm3d::DataLoader::scanDirectory(config.fringe_folder,
                                                 config.pattern.extension);
    std::cout << "Indexed " << all_files.size()
              << " fringe images from: " << config.fringe_folder << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Dataset indexing error: " << e.what() << std::endl;
    return -1;
  }

  // 3. 逐位姿求解绝对相位
  const std::size_t num_freqs = config.pattern.frequencies.size();

  for (int pose = 0; pose < config.pose_count; ++pose) {
    try {
      // 提取当前位姿下的条纹路径二维表: [freq_idx][step_idx]
      const auto pose_paths = msm3d::DataLoader::loadPoseFringePaths(
          all_files, pose, config.pattern);

      std::vector<cv::Mat> wrapped_phases;
      wrapped_phases.reserve(num_freqs);

      // 解算各频率包裹相位
      for (std::size_t f = 0; f < num_freqs; ++f) {
        const auto step_images = msm3d::DataLoader::loadImages(pose_paths[f]);
        wrapped_phases.push_back(
            msm3d::PhaseProcessor::computeWrappedPhase(step_images));
      }

      // 多频外差展开绝对相位
      const cv::Mat abs_phase = msm3d::PhaseProcessor::computeAbsolutePhase(
          wrapped_phases, config.pattern.frequencies);

      // 保存绝对相位（开启无压缩写入，消除 I/O 阻塞）
      const std::string save_path =
          output_dir + "/pose_" + std::to_string(pose) + ".exr";
      if (!msm3d::PhaseProcessor::savePhaseEXR(abs_phase, save_path,
                                               /*compress=*/false)) {
        std::cerr << "Failed to save EXR for pose " << pose << std::endl;
      } else {
        std::cout << "[OK] Pose " << pose
                  << " successfully saved to: " << save_path << std::endl;
      }

    } catch (const std::exception& e) {
      std::cerr << "[FAIL] Error processing pose " << pose << ": " << e.what()
                << std::endl;
    }
  }

  std::cout << "\nAll poses finished successfully." << std::endl;
  return 0;
}
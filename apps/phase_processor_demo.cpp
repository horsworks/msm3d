#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"
#include "msm3d/phase_processor.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/msm_calibration.yaml";

  std::cout << "Loading phase config from: " << config_path << std::endl;
  const auto config = msm3d::loadPhaseConfig(config_path);

  msm3d::DataLoader loader;
  const auto all_files =
      loader.scanDirectory(config.fringe_folder, config.pattern.extension);

  std::cout << "Found " << all_files.size() << " fringe images in directory."
            << std::endl;
  std::filesystem::create_directories(config.output_folder);

  const std::size_t num_freqs = config.pattern.frequencies.size();

  for (int pid = 0; pid < config.pose_count; ++pid) {
    std::cout << "Processing pose " << std::setw(2) << pid << " / "
              << config.pose_count - 1 << " ... " << std::flush;

    const auto pose_paths =
        loader.loadPoseFringePaths(all_files, pid, config.pattern);

    std::vector<cv::Mat> wrapped_phases(num_freqs);
    std::vector<cv::Mat> modulations(num_freqs);

    for (std::size_t f = 0; f < num_freqs; ++f) {
      const auto images = loader.loadImages(pose_paths[f]);
      wrapped_phases[f] =
          msm3d::PhaseProcessor::computeWrappedPhase(images, &modulations[f]);
    }

    // 结合三频调制度进行外差解相 (暗区与黑圆点赋 NaN)
    const cv::Mat abs_phase_raw = msm3d::PhaseProcessor::computeAbsolutePhase(
        wrapped_phases, config.pattern.frequencies, modulations,
        config.min_modulation);

    // 散斑中值滤波抑制激光毛刺
    const cv::Mat clean_phase = msm3d::PhaseProcessor::filterPhaseNoise(
        abs_phase_raw, config.median_filter_size);

    const std::string out_path =
        config.output_folder + "/pose_" + std::to_string(pid) + ".exr";
    msm3d::PhaseProcessor::savePhaseEXR(clean_phase, out_path);

    std::cout << "Done -> " << out_path << std::endl;
  }

  std::cout << "\nStage 1 completed: All clean phase maps generated.\n";
  return 0;
}
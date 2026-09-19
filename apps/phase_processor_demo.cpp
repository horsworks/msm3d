#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"
#include "msm3d/phase_processor.hpp"

#include <algorithm>
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
  std::cout << "Phase fusion: "
            << (config.quality.enable_multifrequency_fusion
                    ? "multi-frequency weighted"
                    : "reference frequency only")
            << std::endl;
  std::cout << "Phase filter: " << config.phase_filter << std::endl;

  std::filesystem::create_directories(config.output_folder);

  const std::size_t num_freqs = config.pattern.frequencies.size();

  for (int pid = 0; pid < config.pose_count; ++pid) {
    std::cout << "Processing pose " << std::setw(2) << pid << " / "
              << config.pose_count - 1 << " ..." << std::endl;

    const auto pose_paths =
        loader.loadPoseFringePaths(all_files, pid, config.pattern);

    std::vector<cv::Mat> wrapped_phases(num_freqs);
    std::vector<cv::Mat> modulations(num_freqs);
    std::vector<cv::Mat> fit_residual_ratios(num_freqs);
    std::vector<cv::Mat> saturation_fractions(num_freqs);

    for (std::size_t f = 0; f < num_freqs; ++f) {
      const auto images = loader.loadImages(pose_paths[f]);
      wrapped_phases[f] = msm3d::PhaseProcessor::computeWrappedPhase(
          images, &modulations[f], &fit_residual_ratios[f],
          &saturation_fractions[f]);
    }

    cv::Mat confidence;
    msm3d::PhaseQualitySummary quality_summary;

    const cv::Mat absolute_phase_raw =
        msm3d::PhaseProcessor::computeAbsolutePhase(
            wrapped_phases, config.pattern.frequencies, modulations,
            fit_residual_ratios, saturation_fractions, config.quality,
            &confidence, &quality_summary);

    cv::Mat clean_phase;
    cv::Mat clean_confidence = confidence.clone();

    if (config.phase_filter == "local_plane") {
      clean_phase = msm3d::PhaseProcessor::filterPhaseLocalPlane(
          absolute_phase_raw, confidence, config.local_plane_filter_size,
          config.local_plane_min_valid_neighbors,
          config.local_plane_robust_scale_rad, &clean_confidence);
    } else if (config.phase_filter == "median") {
      clean_phase = msm3d::PhaseProcessor::filterPhaseNoise(
          absolute_phase_raw, config.median_filter_size);
    } else {
      clean_phase = absolute_phase_raw.clone();
    }

    const std::string phase_path =
        config.output_folder + "/pose_" + std::to_string(pid) + ".exr";

    msm3d::PhaseProcessor::savePhaseEXR(clean_phase, phase_path);

    if (config.save_confidence_map) {
      const std::string confidence_path = config.output_folder +
                                          "/confidence_pose_" +
                                          std::to_string(pid) + ".exr";

      msm3d::PhaseProcessor::savePhaseEXR(clean_confidence, confidence_path);
    }

    const double total =
        static_cast<double>(std::max(quality_summary.total_pixels, 1));

    const double valid_percent =
        100.0 * static_cast<double>(quality_summary.valid_pixels) / total;

    std::cout << "  valid=" << std::fixed << std::setprecision(2)
              << valid_percent << "%"
              << ", reject modulation="
              << 100.0 * quality_summary.rejected_modulation / total << "%"
              << ", fit=" << 100.0 * quality_summary.rejected_fit / total << "%"
              << ", saturation="
              << 100.0 * quality_summary.rejected_saturation / total << "%"
              << ", frequency consistency="
              << 100.0 * quality_summary.rejected_consistency / total << "%"
              << std::endl;

    std::cout << "  mean confidence=" << quality_summary.mean_confidence
              << ", frequency residual p95/p99/max="
              << quality_summary.frequency_consistency_p95_rad << " / "
              << quality_summary.frequency_consistency_p99_rad << " / "
              << quality_summary.max_frequency_consistency_rad << " rad"
              << std::endl;

    if (config.quality.enable_multifrequency_fusion) {
      std::cout << "  fusion disagreement mean/max="
                << quality_summary.mean_fusion_disagreement_rad << " / "
                << quality_summary.max_fusion_disagreement_rad << " rad"
                << std::endl;
    }

    std::cout << "  saved: " << phase_path << std::endl;
  }

  std::cout << "\nPhase maps regenerated.\n";
  return 0;
}

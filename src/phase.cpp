#include "msm3d/phase.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

namespace msm3d {
namespace {

template <typename T>
void readField(const YAML::Node& node, const char* key, T& target) {
  if (node && node[key]) {
    target = node[key].as<T>();
  }
}

}  // namespace

PhaseConfig loadPhaseConfig(const std::string& yaml_file) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_file);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to open/parse phase config: " + yaml_file +
                             " (" + e.what() + ")");
  }

  PhaseConfig config;

  const auto dataset = root["dataset"];
  readField(dataset, "fringe_folder", config.fringe_folder);
  readField(dataset, "pose_count", config.pose_count);
  readField(dataset, "extension", config.extension);

  const auto phase = root["phase"];
  readField(phase, "steps", config.steps);
  if (phase && phase["frequencies"] && phase["frequencies"].IsSequence()) {
    config.frequencies = phase["frequencies"].as<std::vector<int>>();
  }

  const auto output = root["output"];
  readField(output, "phase_folder", config.output_folder);

  if (config.fringe_folder.empty()) {
    throw std::runtime_error("Fringe folder path is empty in config.");
  }
  if (config.steps <= 0) {
    throw std::invalid_argument("Phase shift steps must be positive.");
  }
  if (config.frequencies.empty()) {
    throw std::invalid_argument("Frequencies list must not be empty.");
  }

  return config;
}

PosePhaseImages PhaseProcessor::loadPoseImages(
    int pose_id, const PhaseConfig& config) const {
  std::filesystem::path folder_path(config.fringe_folder);
  if (!std::filesystem::exists(folder_path)) {
    throw std::runtime_error("Fringe folder does not exist: " +
                             config.fringe_folder);
  }

  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      if (config.extension.empty() ||
          entry.path().extension().string() == config.extension) {
        files.push_back(entry.path().string());
      }
    }
  }

  std::sort(files.begin(), files.end());

  const std::size_t num_freqs = config.frequencies.size();
  const std::size_t images_per_pose =
      static_cast<std::size_t>(config.steps) * num_freqs;
  const std::size_t start_idx =
      static_cast<std::size_t>(pose_id) * images_per_pose;
  const std::size_t end_idx = start_idx + images_per_pose;

  if (files.size() < end_idx) {
    throw std::runtime_error("Insufficient images for pose_id " +
                             std::to_string(pose_id) +
                             ". Needed: " + std::to_string(end_idx) +
                             ", Found: " + std::to_string(files.size()));
  }

  PosePhaseImages result;
  result.pose_id = pose_id;
  result.images.resize(num_freqs);

  for (std::size_t f = 0; f < num_freqs; ++f) {
    result.images[f].reserve(config.steps);
    for (int s = 0; s < config.steps; ++s) {
      result.images[f].push_back(files[start_idx + f * config.steps + s]);
    }
  }

  return result;
}

cv::Mat PhaseProcessor::computeWrappedPhase(
    const std::vector<cv::Mat>& images) const {
  if (images.empty()) {
    throw std::invalid_argument("Input images list is empty.");
  }

  const cv::Size img_size = images[0].size();
  cv::Mat sin_sum = cv::Mat::zeros(img_size, CV_64F);
  cv::Mat cos_sum = cv::Mat::zeros(img_size, CV_64F);

  const std::size_t n = images.size();
  const double two_pi_over_n = 2.0 * CV_PI / static_cast<double>(n);

  for (std::size_t i = 0; i < n; ++i) {
    if (images[i].size() != img_size) {
      throw std::invalid_argument("All images must share the same resolution.");
    }

    cv::Mat gray_f64;
    if (images[i].channels() == 1) {
      images[i].convertTo(gray_f64, CV_64F);
    } else {
      cv::Mat gray;
      cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
      gray.convertTo(gray_f64, CV_64F);
    }

    const double delta = two_pi_over_n * static_cast<double>(i);
    sin_sum += gray_f64 * std::sin(delta);
    cos_sum += gray_f64 * std::cos(delta);
  }

  cv::Mat phase;
  cv::phase(cos_sum, sin_sum, phase, false);  // 范围: [0, 2*pi)
  return phase;
}

cv::Mat PhaseProcessor::computeAbsolutePhase(
    const std::vector<cv::Mat>& wrapped,
    const std::vector<int>& frequencies) const {
  if (wrapped.size() != 3 || frequencies.size() != 3) {
    throw std::invalid_argument(
        "Multi-frequency unwrap expects exactly 3 wrapped phases and 3 "
        "frequencies.");
  }

  const cv::Size size = wrapped[0].size();
  for (const auto& mat : wrapped) {
    if (mat.size() != size || mat.type() != CV_64F) {
      throw std::invalid_argument(
          "All wrapped phase images must be identical in size and CV_64F "
          "type.");
    }
  }

  const double f1 = static_cast<double>(frequencies[0]);
  const double f2 = static_cast<double>(frequencies[1]);
  const double f3 = static_cast<double>(frequencies[2]);

  const double T1 = 1.0 / f1;
  const double T2 = 1.0 / f2;
  const double T3 = 1.0 / f3;

  const double T12 = (T1 * T2) / (T2 - T1);
  const double T23 = (T2 * T3) / (T3 - T2);
  const double T123 = (T12 * T23) / (T23 - T12);

  const double R12 = T12 / T1;
  const double R123 = T123 / T12;
  constexpr double kTwoPi = 2.0 * CV_PI;

  cv::Mat result(size, CV_64F);

  for (int y = 0; y < size.height; ++y) {
    const double* p1_ptr = wrapped[0].ptr<double>(y);
    const double* p2_ptr = wrapped[1].ptr<double>(y);
    const double* p3_ptr = wrapped[2].ptr<double>(y);
    double* out_ptr = result.ptr<double>(y);

    for (int x = 0; x < size.width; ++x) {
      const double p1 = p1_ptr[x];
      const double p2 = p2_ptr[x];
      const double p3 = p3_ptr[x];

      double p12 = p1 - p2;
      if (p12 < 0.0) p12 += kTwoPi;

      double p23 = p2 - p3;
      if (p23 < 0.0) p23 += kTwoPi;

      double p123 = p12 - p23;
      if (p123 < 0.0) p123 += kTwoPi;

      const double abs12 =
          p12 + std::round((p123 * R123 - p12) / kTwoPi) * kTwoPi;
      const double k = std::round((abs12 * R12 - p1) / kTwoPi);

      out_ptr[x] = p1 + k * kTwoPi;
    }
  }

  return result;
}

bool PhaseProcessor::savePhaseEXR(const cv::Mat& phase,
                                  const std::string& filename) const {
  if (phase.empty()) {
    return false;
  }
  cv::Mat out;
  phase.convertTo(out, CV_32F);
  return cv::imwrite(filename, out);
}

}  // namespace msm3d

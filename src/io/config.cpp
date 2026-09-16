#include "msm3d/io/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace msm3d {
namespace {

template <typename T>
void readField(const YAML::Node& node, const char* key, T& target) {
  if (node && node[key]) {
    target = node[key].as<T>();
  }
}

inline void readField(const YAML::Node& node, const char* key, bool& target) {
  if (node && node[key]) {
    try {
      target = node[key].as<bool>();
    } catch (const YAML::BadConversion&) {
      target = (node[key].as<int>() != 0);
    }
  }
}

CalibrationPattern parsePattern(const std::string& pattern) {
  if (pattern == "chessboard") {
    return CalibrationPattern::kChessboard;
  }
  if (pattern == "circles") {
    return CalibrationPattern::kSymmetricCircles;
  }
  if (pattern == "asymmetric_circles") {
    return CalibrationPattern::kAsymmetricCircles;
  }
  throw std::invalid_argument("Unknown calibration pattern: " + pattern);
}

void validateBoardConfig(const CalibrationBoard& board) {
  if (board.columns <= 1 || board.rows <= 1) {
    throw std::invalid_argument(
        "Calibration board rows and columns must be greater than 1.");
  }
  if (!std::isfinite(board.spacing) || board.spacing <= 0.0) {
    throw std::invalid_argument(
        "Calibration board spacing must be positive and finite.");
  }
}

}  // namespace

CameraCalibrationConfig loadCameraCalibrationConfig(
    const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(
        "Failed to open/parse camera calibration config: " + config_path +
        " (" + e.what() + ")");
  }

  CameraCalibrationConfig config;

  // input
  const auto input_node = root["input"];
  readField(input_node, "image_dir", config.image_dir);

  // board
  const auto board_node = root["board"];
  if (board_node && board_node["pattern"]) {
    config.board.pattern =
        parsePattern(board_node["pattern"].as<std::string>());
  }
  readField(board_node, "columns", config.board.columns);
  readField(board_node, "rows", config.board.rows);
  readField(board_node, "spacing", config.board.spacing);

  // circle_detector
  const auto detector_node = root["circle_detector"];
  if (detector_node) {
    readField(detector_node, "min_threshold",
              config.circle_detector.min_threshold);
    readField(detector_node, "max_threshold",
              config.circle_detector.max_threshold);
    readField(detector_node, "threshold_step",
              config.circle_detector.threshold_step);

    if (detector_node["min_repeatability"]) {
      int rep = detector_node["min_repeatability"].as<int>();
      config.circle_detector.min_repeatability =
          static_cast<std::size_t>(std::max(rep, 1));
    }

    readField(detector_node, "min_dist_between_blobs",
              config.circle_detector.min_dist_between_blobs);
    readField(detector_node, "filter_by_area",
              config.circle_detector.filter_by_area);
    readField(detector_node, "min_area", config.circle_detector.min_area);
    readField(detector_node, "max_area", config.circle_detector.max_area);
    readField(detector_node, "filter_by_circularity",
              config.circle_detector.filter_by_circularity);
    readField(detector_node, "min_circularity",
              config.circle_detector.min_circularity);
    readField(detector_node, "filter_by_convexity",
              config.circle_detector.filter_by_convexity);
    readField(detector_node, "min_convexity",
              config.circle_detector.min_convexity);
    readField(detector_node, "filter_by_inertia",
              config.circle_detector.filter_by_inertia);
    readField(detector_node, "min_inertia_ratio",
              config.circle_detector.min_inertia_ratio);
    readField(detector_node, "filter_by_color",
              config.circle_detector.filter_by_color);

    if (detector_node["blob_color"]) {
      int color = detector_node["blob_color"].as<int>();
      config.circle_detector.blob_color =
          static_cast<unsigned char>(std::clamp(color, 0, 255));
    }
  }

  // calibration
  const auto calibration_node = root["calibration"];
  readField(calibration_node, "fix_k3", config.calibration.fix_k3);

  // output
  const auto output_node = root["output"];
  readField(output_node, "result_file", config.result_file);
  readField(output_node, "detection_dir", config.detection_dir);
  readField(output_node, "blob_dir", config.blob_dir);
  readField(output_node, "save_detection_debug", config.save_detection_debug);
  readField(output_node, "save_blob_debug", config.save_blob_debug);

  validateBoardConfig(config.board);

  if (config.image_dir.empty()) {
    throw std::runtime_error("Camera calibration image directory is empty.");
  }
  if (config.result_file.empty()) {
    throw std::runtime_error("Camera calibration result file is empty.");
  }

  return config;
}

PhaseConfig loadPhaseConfig(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to open/parse phase config: " +
                             config_path + " (" + e.what() + ")");
  }

  PhaseConfig config;

  const auto dataset = root["dataset"];
  readField(dataset, "fringe_folder", config.fringe_folder);
  readField(dataset, "pose_count", config.pose_count);
  readField(dataset, "extension", config.pattern.extension);

  const auto phase = root["phase"];
  readField(phase, "steps", config.pattern.steps);
  if (phase && phase["frequencies"] && phase["frequencies"].IsSequence()) {
    config.pattern.frequencies = phase["frequencies"].as<std::vector<int>>();
  }

  const auto output = root["output"];
  readField(output, "phase_folder", config.output_folder);

  if (config.fringe_folder.empty()) {
    throw std::runtime_error("Fringe folder path is empty in config.");
  }
  if (config.pattern.steps <= 0) {
    throw std::invalid_argument("Phase shift steps must be positive.");
  }
  if (config.pattern.frequencies.empty()) {
    throw std::invalid_argument("Frequencies list must not be empty.");
  }

  return config;
}

}  // namespace msm3d
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

  readField(phase, "min_modulation", config.min_modulation);
  readField(phase, "median_filter_size", config.median_filter_size);

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

CameraCalibrationResult loadCameraCalibrationResult(
    const std::string& result_path) {
  cv::FileStorage fs(result_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open camera calibration result file: " +
                             result_path);
  }

  CameraCalibrationResult result;
  fs["camera_matrix"] >> result.camera_matrix;
  fs["distortion_coefficients"] >> result.distortion_coefficients;
  fs["rms"] >> result.rms;
  fs["reprojection_error"] >> result.reprojection_error;

  const cv::FileNode rvecs_node = fs["rotation_vectors"];
  if (rvecs_node.isSeq()) {
    for (const auto& node : rvecs_node) {
      cv::Mat rvec;
      node >> rvec;
      result.rotation_vectors.push_back(rvec);
    }
  }

  const cv::FileNode tvecs_node = fs["translation_vectors"];
  if (tvecs_node.isSeq()) {
    for (const auto& node : tvecs_node) {
      cv::Mat tvec;
      node >> tvec;
      result.translation_vectors.push_back(tvec);
    }
  }

  if (result.camera_matrix.empty() || result.distortion_coefficients.empty()) {
    throw std::runtime_error(
        "Invalid camera calibration file: missing camera_matrix or "
        "distortion_coefficients.");
  }

  return result;
}

MsmCalibrationConfig loadMsmCalibrationConfig(const std::string& config_path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(config_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to open/parse MSM calibration config: " +
                             config_path + " (" + e.what() + ")");
  }

  MsmCalibrationConfig config;

  const auto dataset = root["dataset"];
  readField(dataset, "camera_param_file", config.camera_param_file);
  if (dataset && dataset["train_poses"] &&
      dataset["train_poses"].IsSequence()) {
    config.train_poses = dataset["train_poses"].as<std::vector<int>>();
  }
  if (dataset && dataset["test_poses"] && dataset["test_poses"].IsSequence()) {
    config.test_poses = dataset["test_poses"].as<std::vector<int>>();
  }

  const auto msm = root["msm_calibration"];
  readField(msm, "auto_phase_range", config.options.auto_phase_range);
  readField(msm, "min_covisible_poses", config.options.min_covisible_poses);
  readField(msm, "min_psi", config.options.min_psi);
  readField(msm, "max_psi", config.options.max_psi);
  readField(msm, "plane_count", config.options.plane_count);
  readField(msm, "harmonic_order", config.options.harmonic_order);
  readField(msm, "min_spread_ratio", config.options.min_spread_ratio);
  readField(msm, "max_thickness_ratio", config.options.max_thickness_ratio);
  readField(msm, "max_plane_rms_mm", config.options.max_plane_rms_mm);

  const auto output = root["output"];
  readField(output, "result_file", config.result_file);
  readField(output, "phase_folder", config.phase_folder);

  if (config.camera_param_file.empty()) {
    throw std::runtime_error("camera_param_file is required in config.");
  }
  if (config.phase_folder.empty()) {
    throw std::runtime_error("phase_folder is required in config.");
  }
  if (config.train_poses.empty()) {
    throw std::invalid_argument("train_poses list must not be empty.");
  }
  if (config.options.min_psi >= config.options.max_psi) {
    throw std::invalid_argument("min_psi must be strictly less than max_psi.");
  }
  if (config.options.plane_count < 10) {
    throw std::invalid_argument("plane_count must be at least 10.");
  }
  if (config.options.harmonic_order < 0 || config.options.harmonic_order > 2) {
    throw std::invalid_argument("harmonic_order must be 0, 1, or 2.");
  }

  return config;
}

}  // namespace msm3d
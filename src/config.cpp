#include "msm3d/config.hpp"

#include <yaml-cpp/yaml.h>

namespace msm3d {

Config loadConfig(const std::string& filename) {
  Config config;

  YAML::Node root = YAML::LoadFile(filename);

  config.board_width = root["camera"]["board_width"].as<int>();

  config.board_height = root["camera"]["board_height"].as<int>();

  config.square_size_mm = root["camera"]["square_size_mm"].as<double>();

  config.camera_image_folder = root["camera"]["image_folder"].as<std::string>();

  config.phase_root_folder = root["phase"]["root_folder"].as<std::string>();

  config.phase_steps = root["phase"]["steps"].as<int>();

  for (const auto& item : root["phase"]["frequencies"]) {
    config.frequencies.push_back(item.as<int>());
  }

  return config;
}

}  // namespace msm3d

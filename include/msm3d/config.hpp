#pragma once

#include <string>
#include <vector>

namespace msm3d {

struct Config {
  int board_width = 9;
  int board_height = 6;
  double square_size_mm = 10.0;

  std::string camera_image_folder;

  std::string phase_root_folder;
  std::vector<int> frequencies;
  int phase_steps = 24;
};

Config loadConfig(const std::string& filename);

}  // namespace msm3d

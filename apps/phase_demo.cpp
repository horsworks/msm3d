#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "msm3d/phase.hpp"

int main() {
  const std::string input_dir = "data/fringe";
  const std::string output_dir = "output";

  std::vector<cv::String> file_names;
  cv::glob(input_dir + "/*.bmp", file_names);

  if (file_names.empty()) {
    std::cerr << "No images found in the input directory: " << input_dir
              << std::endl;
    return -1;
  }

  std::sort(file_names.begin(), file_names.end());

  std::vector<cv::Mat> images;
  for (const auto& file_name : file_names) {
    cv::Mat img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
    if (!img.empty()) {
      images.push_back(img);
    } else {
      std::cerr << "Failed to read image: " << file_name << std::endl;
      return -1;
    }
  }

  if (images.size() < 3) {
    std::cerr << "At least 3 images are required for phase computation."
              << std::endl;
    return -1;
  }

  msm3d::PhaseProcessor processor;
  msm3d::PhaseResult result = processor.computePhase(images);

  if (result.absolute_phase.empty() || result.confidence.empty() ||
      result.valid_mask.empty()) {
    std::cerr << "Phase computation failed." << std::endl;
    return -1;
  }

  auto saveFloatMatAsImage = [&](const cv::Mat& mat,
                                 const std::string& filename) {
    if (mat.empty()) return;
    cv::Mat normalized;
    cv::normalize(mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    cv::imwrite(output_dir + "/" + filename, normalized);
  };

  saveFloatMatAsImage(result.absolute_phase, "absolute_phase.bmp");
  saveFloatMatAsImage(result.confidence, "confidence.bmp");
  saveFloatMatAsImage(result.valid_mask, "valid_mask.bmp");

  std::cout << "Done! Results saved sucessfully.\n";

  return 0;
}
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace msm3d {

void printPlaneQualityDiagnostics(const std::vector<DiscretePlane>& planes,
                                  const MsmCalibrationOptions& options) {
  int valid_count = 0;
  int rejected_pose_count = 0;
  int rejected_spread = 0;
  int rejected_thickness = 0;
  int rejected_rms = 0;

  std::vector<double> valid_rms;
  std::vector<double> valid_spread;
  std::vector<double> valid_thickness;
  std::vector<double> valid_confidence;

  for (const auto& plane : planes) {
    if (plane.pose_count < options.min_covisible_poses) {
      ++rejected_pose_count;
    }
    if (plane.spread_ratio < options.min_spread_ratio) {
      ++rejected_spread;
    }
    if (plane.thickness_ratio > options.max_thickness_ratio) {
      ++rejected_thickness;
    }
    if (plane.rms_mm > options.max_plane_rms_mm) {
      ++rejected_rms;
    }

    if (!plane.valid) {
      continue;
    }

    ++valid_count;
    valid_rms.push_back(plane.rms_mm);
    valid_spread.push_back(plane.spread_ratio);
    valid_thickness.push_back(plane.thickness_ratio);
    valid_confidence.push_back(plane.confidence);
  }

  auto print_stats = [](const char* name, std::vector<double> values) {
    if (values.empty()) {
      std::cout << "  " << name << ": n/a" << std::endl;
      return;
    }

    std::sort(values.begin(), values.end());
    const double min_v = values.front();
    const double med_v = values[values.size() / 2];
    const double max_v = values.back();

    std::cout << "  " << name << ": min=" << min_v << ", median=" << med_v
              << ", max=" << max_v << std::endl;
  };

  std::cout << "\nPlane quality diagnostics" << std::endl;
  std::cout << "  valid planes: " << valid_count << " / " << planes.size()
            << std::endl;
  print_stats("RMS [mm]", valid_rms);
  print_stats("spread ratio sigma2/sigma1", valid_spread);
  print_stats("thickness ratio sigma3/sigma2", valid_thickness);
  print_stats("normalized confidence", valid_confidence);

  std::cout << "  rejection counters (a plane may hit multiple rules):"
            << std::endl;
  std::cout << "    insufficient poses: " << rejected_pose_count << std::endl;
  std::cout << "    insufficient spread: " << rejected_spread << std::endl;
  std::cout << "    excessive thickness: " << rejected_thickness << std::endl;
  std::cout << "    excessive RMS: " << rejected_rms << std::endl;
}
void printPosePlaneResidualDiagnostics(const char* title,
                                       const std::vector<DiscretePlane>& planes,
                                       const PlaneObservationSets& observations,
                                       const std::vector<int>& pose_ids) {
  if (planes.size() != observations.size() || pose_ids.empty()) {
    return;
  }

  std::vector<double> sum_plane_rms_sq(pose_ids.size(), 0.0);
  std::vector<double> sum_plane_bias(pose_ids.size(), 0.0);
  std::vector<int> valid_plane_counts(pose_ids.size(), 0);

  for (std::size_t p = 0; p < planes.size(); ++p) {
    if (!planes[p].valid) {
      continue;
    }

    const auto& groups = observations[p];
    for (std::size_t pose_idx = 0;
         pose_idx < pose_ids.size() && pose_idx < groups.size(); ++pose_idx) {
      const auto& points = groups[pose_idx];
      if (points.empty()) {
        continue;
      }

      double sum_sq = 0.0;
      double sum_signed = 0.0;

      for (const auto& point : points) {
        const double residual = planes[p].normal.dot(point) + planes[p].d;
        sum_sq += residual * residual;
        sum_signed += residual;
      }

      const double count = static_cast<double>(points.size());
      const double group_rms = std::sqrt(sum_sq / count);
      const double group_bias = sum_signed / count;

      sum_plane_rms_sq[pose_idx] += group_rms * group_rms;
      sum_plane_bias[pose_idx] += group_bias;
      ++valid_plane_counts[pose_idx];
    }
  }

  std::cout << "\n" << title << std::endl;
  for (std::size_t i = 0; i < pose_ids.size(); ++i) {
    if (valid_plane_counts[i] <= 0) {
      std::cout << "  pose " << std::setw(2) << pose_ids[i] << ": n/a"
                << std::endl;
      continue;
    }

    const double plane_count = static_cast<double>(valid_plane_counts[i]);
    const double rms = std::sqrt(sum_plane_rms_sq[i] / plane_count);
    const double mean_bias = sum_plane_bias[i] / plane_count;

    std::cout << "  pose " << std::setw(2) << pose_ids[i]
              << ": plane RMSE=" << std::fixed << std::setprecision(5) << rms
              << " mm, mean signed bias=" << mean_bias
              << " mm, planes=" << valid_plane_counts[i] << std::endl;
  }
}
void printPhaseBinDiagnostics(const ReconstructionDiagnostics& diagnostics,
                              double min_psi, double max_psi) {
  const std::size_t bin_count = diagnostics.phase_bin_sample_count.size();
  if (bin_count == 0 || !(max_psi > min_psi)) {
    return;
  }

  std::cout << "    phase-bin RMSE [psi range: 3D / plane, samples]"
            << std::endl;

  for (std::size_t bin = 0; bin < bin_count; ++bin) {
    const double left = min_psi + (max_psi - min_psi) *
                                      static_cast<double>(bin) /
                                      static_cast<double>(bin_count);
    const double right = min_psi + (max_psi - min_psi) *
                                       static_cast<double>(bin + 1) /
                                       static_cast<double>(bin_count);

    std::cout << "      [" << std::fixed << std::setprecision(1) << left << ", "
              << right << "]: ";

    if (diagnostics.phase_bin_sample_count[bin] <= 0) {
      std::cout << "n/a" << std::endl;
      continue;
    }

    std::cout << std::setprecision(5) << diagnostics.phase_bin_rmse_3d_mm[bin]
              << " / " << diagnostics.phase_bin_plane_rmse_mm[bin] << " mm, "
              << diagnostics.phase_bin_sample_count[bin] << std::endl;
  }
}

double computeHarmonicPlaneOffsetRmse(
    const MsmCalibrationResult& result,
    const std::vector<DiscretePlane>& reference_planes,
    double* out_max_abs_mm) {
  double weighted_sum_sq = 0.0;
  double weight_sum = 0.0;
  double max_abs_mm = 0.0;

  for (const auto& plane : reference_planes) {
    if (!plane.valid) {
      continue;
    }

    cv::Vec3d model_normal;
    double model_d = 0.0;
    result.evaluatePlane(plane.psi, model_normal, model_d);

    if (model_normal.dot(plane.normal) < 0.0) {
      model_normal = -model_normal;
      model_d = -model_d;
    }

    // reference_planes already use the continuous normal model, so d
    // difference is the orthogonal plane offset error.
    const double residual = model_d - plane.d;
    const double weight = std::max(plane.confidence, 1e-8);

    weighted_sum_sq += weight * residual * residual;
    weight_sum += weight;
    max_abs_mm = std::max(max_abs_mm, std::abs(residual));
  }

  if (out_max_abs_mm != nullptr) {
    *out_max_abs_mm = max_abs_mm;
  }

  return (weight_sum > 0.0) ? std::sqrt(weighted_sum_sq / weight_sum) : 0.0;
}

void printNormalAxisDiagnostics(const std::vector<DiscretePlane>& planes,
                                const cv::Vec3d& axis) {
  double weighted_sum_sq = 0.0;
  double weight_sum = 0.0;
  double max_abs = 0.0;

  for (const auto& plane : planes) {
    if (!plane.valid) {
      continue;
    }

    const double axial_component = plane.normal.dot(axis);
    const double weight = std::max(plane.confidence, 1e-8);
    weighted_sum_sq += weight * axial_component * axial_component;
    weight_sum += weight;
    max_abs = std::max(max_abs, std::abs(axial_component));
  }

  const double rms =
      (weight_sum > 0.0) ? std::sqrt(weighted_sum_sq / weight_sum) : 0.0;

  std::cout << "  observed normal axial-component RMS: " << rms
            << ", max: " << max_abs << std::endl;
}

}  // namespace msm3d

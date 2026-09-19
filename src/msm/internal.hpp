#pragma once

#include "msm3d/msm/model.hpp"
#include "msm3d/msm/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace msm3d {

inline constexpr double kEpsilon = 1e-12;

using PosePointGroups = std::vector<std::vector<cv::Vec3d>>;
using PlaneObservationSets = std::vector<PosePointGroups>;

struct ReconstructionDiagnostics {
  double rmse_3d_mm = 0.0;
  double model_plane_rmse_mm = 0.0;
  double ray_plane_denom_min = 0.0;
  double ray_plane_denom_p05 = 0.0;
  double ray_plane_denom_median = 0.0;
  double amplification = 0.0;
  int sample_count = 0;

  std::vector<double> phase_bin_rmse_3d_mm;
  std::vector<double> phase_bin_plane_rmse_mm;
  std::vector<int> phase_bin_sample_count;
};

void normalizePlaneConfidences(std::vector<DiscretePlane>& planes);

DiscretePlane fitOffsetForFixedNormal(const PosePointGroups& points_by_pose,
                                      double psi,
                                      const cv::Vec3d& fixed_normal,
                                      const DiscretePlane& geometry_source,
                                      const MsmCalibrationOptions& options);

DiscretePlane fitPlaneRobustTLS(const PosePointGroups& points_by_pose,
                                double psi,
                                const MsmCalibrationOptions& options);

void enforceNormalConsistency(std::vector<DiscretePlane>& planes);

cv::Vec3d rotateAroundAxis(const cv::Vec3d& vector, const cv::Vec3d& axis,
                           double angle_rad);

double invertAngleModel(const RationalAngleModel& model, double target_theta,
                        double min_psi, double max_psi);

std::vector<double> makeUniformSamples(double min_value, double max_value,
                                       int count);

double computeAngleModelRmse(const std::vector<DiscretePlane>& planes,
                             const std::vector<double>& thetas,
                             const RationalAngleModel& model,
                             double* out_max_abs_error = nullptr);

bool solveCenterFromPlanes(const std::vector<DiscretePlane>& planes,
                           const cv::Vec3d& u, const cv::Vec3d& v,
                           cv::Vec3d& out_center);

bool solveNominalRotationGeometry(const std::vector<DiscretePlane>& planes,
                                  double ref_psi, cv::Vec3d& out_w,
                                  cv::Vec3d& out_S0, cv::Vec3d& out_n0,
                                  cv::Vec3d& out_u, cv::Vec3d& out_v);

std::vector<double> computeRelativeAngles(
    const std::vector<DiscretePlane>& planes, const cv::Vec3d& w,
    const cv::Vec3d& n0, bool enforce_monotonic = true);

RationalAngleModel fitRationalAngleModel(
    const std::vector<DiscretePlane>& planes,
    const std::vector<double>& thetas, double psi_ref);

HarmonicDriftModel fitHarmonicDriftModel(
    const std::vector<DiscretePlane>& planes,
    const std::vector<double>& thetas, const cv::Vec3d& w,
    const cv::Vec3d& initial_s0, const cv::Vec3d& u, const cv::Vec3d& v,
    cv::Vec3d& out_reference_s0, int order = 1,
    double regularization = 1e-3);

ReconstructionDiagnostics evaluateReconstructionDetailed(
    const MsmCalibrationResult& result, const cv::Mat& phase_map,
    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
    const BoardPlane& board_plane, double min_psi, double max_psi,
    int pixel_stride, int phase_bin_count = 0);

void printPlaneQualityDiagnostics(const std::vector<DiscretePlane>& planes,
                                  const MsmCalibrationOptions& options);

void printPosePlaneResidualDiagnostics(
    const char* title, const std::vector<DiscretePlane>& planes,
    const PlaneObservationSets& observations,
    const std::vector<int>& pose_ids);

void printPhaseBinDiagnostics(const ReconstructionDiagnostics& diagnostics,
                              double min_psi, double max_psi);

double computeHarmonicPlaneOffsetRmse(
    const MsmCalibrationResult& result,
    const std::vector<DiscretePlane>& reference_planes,
    double* out_max_abs_mm = nullptr);

void printNormalAxisDiagnostics(const std::vector<DiscretePlane>& planes,
                                const cv::Vec3d& axis);

}  // namespace msm3d

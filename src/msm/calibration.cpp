#include "msm3d/msm/calibration.hpp"

#include "msm3d/msm/iso_phase.hpp"
#include "msm3d/msm/reconstruction.hpp"
#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace msm3d {

MsmCalibrationResult calibrateMsm(const MsmCalibrationConfig& config,
                                  const CameraCalibrationResult& camera_calib,
                                  const std::vector<cv::Mat>& all_phase_maps) {
  std::cout << "\nStarting MSM calibration pipeline..." << std::endl;

  const std::size_t train_count = config.train_poses.size();
  std::vector<BoardPlane> train_board_planes;
  std::vector<cv::Mat> train_phase_maps;

  train_board_planes.reserve(train_count);
  train_phase_maps.reserve(train_count);

  std::cout << "[Step 1] Preparing board planes for " << train_count
            << " training poses..." << std::endl;

  for (std::size_t i = 0; i < train_count; ++i) {
    const int pid = config.train_poses[i];

    if (pid < 0 ||
        pid >= static_cast<int>(camera_calib.rotation_vectors.size()) ||
        pid >= static_cast<int>(camera_calib.translation_vectors.size()) ||
        pid >= static_cast<int>(all_phase_maps.size()) ||
        all_phase_maps[static_cast<std::size_t>(pid)].empty()) {
      throw std::runtime_error("Invalid or missing training pose: " +
                               std::to_string(pid));
    }

    const auto board_plane =
        computeBoardPlane(camera_calib.rotation_vectors[pid],
                          camera_calib.translation_vectors[pid]);

    train_board_planes.push_back(board_plane);
    train_phase_maps.push_back(all_phase_maps[static_cast<std::size_t>(pid)]);

    std::cout << "  Pose " << std::setw(2) << pid << ": board normal=["
              << board_plane.normal[0] << ", " << board_plane.normal[1] << ", "
              << board_plane.normal[2] << "], distance d=" << board_plane.d
              << " mm" << std::endl;
  }

  double eff_min_psi = config.options.min_psi;
  double eff_max_psi = config.options.max_psi;

  if (config.options.auto_phase_range) {
    std::cout << "[Step 2] Auto-detecting multi-view co-visible phase range "
              << "(min_covisible_poses=" << config.options.min_covisible_poses
              << ") ..." << std::endl;

    const auto auto_range = autoDetectValidPhaseRange(
        train_phase_maps, config.options.min_covisible_poses);
    eff_min_psi = auto_range.first;
    eff_max_psi = auto_range.second;

    std::cout << "  Auto-detected co-visible range: [" << eff_min_psi << ", "
              << eff_max_psi << "] rad (span: " << eff_max_psi - eff_min_psi
              << " rad)" << std::endl;
  } else {
    std::cout << "[Step 2] Using configured phase range: [" << eff_min_psi
              << ", " << eff_max_psi << "] rad" << std::endl;
  }

  if (!(eff_max_psi > eff_min_psi)) {
    throw std::runtime_error(
        "Invalid effective phase range after range detection.");
  }

  auto collect_points_by_pose = [&](double target_psi) {
    PosePointGroups points_by_pose(train_count);

    for (std::size_t i = 0; i < train_count; ++i) {
      const auto subpixels =
          extractIsoPhaseSubpixels(train_phase_maps[i], target_psi, cv::Mat(),
                                   config.options.min_local_phase_gradient,
                                   config.options.iso_fit_half_window,
                                   config.options.max_local_phase_gradient,
                                   config.options.max_lateral_jump_px);

      if (subpixels.empty()) {
        continue;
      }

      auto points =
          projectSubpixelsToBoard(subpixels, camera_calib.camera_matrix,
                                  camera_calib.distortion_coefficients,
                                  train_board_planes[i], 120.0, 250.0);

      if (static_cast<int>(points.size()) <
          config.options.min_points_per_pose) {
        continue;
      }

      points_by_pose[i] = std::move(points);
    }

    return points_by_pose;
  };

  auto fit_plane_set = [&](const std::vector<double>& target_phases,
                           bool print_examples,
                           PlaneObservationSets* out_observations,
                           int required_pose_count) {
    std::vector<DiscretePlane> planes;
    planes.reserve(target_phases.size());

    if (out_observations != nullptr) {
      out_observations->clear();
      out_observations->reserve(target_phases.size());
    }

    auto fit_options = config.options;
    fit_options.min_covisible_poses = std::max(1, required_pose_count);

    for (std::size_t p_idx = 0; p_idx < target_phases.size(); ++p_idx) {
      const double target_psi = target_phases[p_idx];
      auto points_by_pose = collect_points_by_pose(target_psi);

      auto plane = fitPlaneRobustTLS(points_by_pose, target_psi, fit_options);
      planes.push_back(plane);

      if (out_observations != nullptr) {
        out_observations->push_back(std::move(points_by_pose));
      }

      if (print_examples && (p_idx == 0 || p_idx == target_phases.size() / 2 ||
                             p_idx + 1 == target_phases.size())) {
        std::cout << "  Plane " << std::setw(2) << p_idx
                  << " (psi=" << std::fixed << std::setprecision(2)
                  << target_psi << "): poses=" << plane.pose_count
                  << ", pts=" << plane.point_count
                  << ", rms=" << std::setprecision(5) << plane.rms_mm << " mm"
                  << ", spread=" << plane.spread_ratio
                  << ", thickness=" << plane.thickness_ratio
                  << ", valid=" << (plane.valid ? "YES" : "NO") << std::endl;
      }
    }

    enforceNormalConsistency(planes);
    normalizePlaneConfidences(planes);
    return planes;
  };

  const double ref_psi = 0.5 * (eff_min_psi + eff_max_psi);
  const int final_plane_count = std::max(10, config.options.plane_count);
  const int coarse_plane_count = std::clamp(final_plane_count / 2, 24, 40);

  std::cout << "[Step 3] Estimating phase-to-angle mapping..." << std::endl;

  std::vector<DiscretePlane> coarse_planes;
  double coarse_min_psi = eff_min_psi;
  double coarse_max_psi = eff_max_psi;
  int coarse_required_poses = config.options.min_covisible_poses;
  bool initialization_found = false;

  const int maximum_initial_pose_count = static_cast<int>(train_count);
  const int minimum_initial_pose_count = std::max(
      2,
      std::min(config.options.min_covisible_poses, maximum_initial_pose_count));

  for (int required_poses = maximum_initial_pose_count;
       required_poses >= minimum_initial_pose_count; --required_poses) {
    const auto candidate_range =
        autoDetectValidPhaseRange(train_phase_maps, required_poses);

    if (!(candidate_range.second > candidate_range.first)) {
      continue;
    }

    const auto candidate_phases = makeUniformSamples(
        candidate_range.first, candidate_range.second, coarse_plane_count);

    auto candidate_planes =
        fit_plane_set(candidate_phases, false, nullptr, required_poses);

    int valid_count = 0;
    for (const auto& plane : candidate_planes) {
      if (plane.valid) {
        ++valid_count;
      }
    }

    std::cout << "  initialization candidate: " << required_poses << "/"
              << train_count << " co-visible poses, range=[" << std::fixed
              << std::setprecision(3) << candidate_range.first << ", "
              << candidate_range.second << "] rad"
              << ", valid planes=" << valid_count << "/" << coarse_plane_count
              << std::endl;

    if (valid_count >= 6) {
      coarse_planes = std::move(candidate_planes);
      coarse_min_psi = candidate_range.first;
      coarse_max_psi = candidate_range.second;
      coarse_required_poses = required_poses;
      initialization_found = true;
      break;
    }
  }

  if (!initialization_found) {
    std::cout << "  No stable initialization range was found. "
              << "Plane diagnostics for the configured overlap are:"
              << std::endl;

    const auto fallback_phases =
        makeUniformSamples(eff_min_psi, eff_max_psi, coarse_plane_count);
    auto fallback_planes = fit_plane_set(fallback_phases, false, nullptr,
                                         config.options.min_covisible_poses);
    printPlaneQualityDiagnostics(fallback_planes, config.options);

    throw std::runtime_error(
        "Insufficient valid initial planes for angle estimation.");
  }

  std::cout << "  selected initialization range: [" << coarse_min_psi << ", "
            << coarse_max_psi << "] rad using " << coarse_required_poses << "/"
            << train_count << " co-visible poses" << std::endl;

  const double coarse_ref_psi = 0.5 * (coarse_min_psi + coarse_max_psi);

  cv::Vec3d coarse_w;
  cv::Vec3d coarse_s0;
  cv::Vec3d coarse_n0;
  cv::Vec3d coarse_u;
  cv::Vec3d coarse_v;

  if (!solveNominalRotationGeometry(coarse_planes, coarse_ref_psi, coarse_w,
                                    coarse_s0, coarse_n0, coarse_u, coarse_v)) {
    throw std::runtime_error("Failed to solve initial rotation geometry.");
  }

  const auto coarse_thetas =
      computeRelativeAngles(coarse_planes, coarse_w, coarse_n0, true);
  const auto coarse_angle_model =
      fitRationalAngleModel(coarse_planes, coarse_thetas, coarse_ref_psi);

  if (!coarse_angle_model.valid) {
    throw std::runtime_error(
        "Failed to estimate initial phase-to-angle model.");
  }

  RationalAngleModel sampling_model = coarse_angle_model;
  MsmCalibrationResult result;
  std::vector<DiscretePlane> discrete_planes;
  PlaneObservationSets final_observations;
  std::vector<double> thetas;
  std::vector<double> previous_phases;

  const int max_resampling_iterations =
      std::max(1, config.options.angle_resampling_iterations);

  std::cout << "[Step 4] Fitting " << final_plane_count
            << " planes uniformly in optical angle..." << std::endl;

  for (int iteration = 0; iteration < max_resampling_iterations; ++iteration) {
    const double current_theta_min = sampling_model.evaluate(eff_min_psi);
    const double current_theta_max = sampling_model.evaluate(eff_max_psi);

    if (!std::isfinite(current_theta_min) ||
        !std::isfinite(current_theta_max) ||
        std::abs(current_theta_max - current_theta_min) < 1e-8) {
      throw std::runtime_error(
          "Invalid optical-angle range during resampling.");
    }

    const auto target_thetas = makeUniformSamples(
        current_theta_min, current_theta_max, final_plane_count);

    std::vector<double> refined_phases;
    refined_phases.reserve(target_thetas.size());

    for (const double theta : target_thetas) {
      const double psi =
          invertAngleModel(sampling_model, theta, eff_min_psi, eff_max_psi);
      if (!std::isfinite(psi)) {
        throw std::runtime_error(
            "Failed to invert phase-to-angle model during "
            "uniform-angle resampling.");
      }
      refined_phases.push_back(psi);
    }

    double max_phase_update = 0.0;
    if (!previous_phases.empty() &&
        previous_phases.size() == refined_phases.size()) {
      for (std::size_t i = 0; i < refined_phases.size(); ++i) {
        max_phase_update = std::max(
            max_phase_update, std::abs(refined_phases[i] - previous_phases[i]));
      }
    }

    PlaneObservationSets observations;
    auto planes = fit_plane_set(
        refined_phases, iteration + 1 == max_resampling_iterations,
        &observations, config.options.min_covisible_poses);

    int valid_count = 0;
    for (const auto& plane : planes) {
      if (plane.valid) {
        ++valid_count;
      }
    }

    if (valid_count < 6) {
      throw std::runtime_error(
          "Insufficient valid planes for MSM calibration.");
    }

    MsmCalibrationResult iteration_result;
    if (!solveNominalRotationGeometry(
            planes, ref_psi, iteration_result.nominal_axis_w,
            iteration_result.nominal_center_s0, iteration_result.ref_normal_n0,
            iteration_result.basis_u, iteration_result.basis_v)) {
      throw std::runtime_error("Failed to solve nominal rotation geometry.");
    }

    auto iteration_thetas =
        computeRelativeAngles(planes, iteration_result.nominal_axis_w,
                              iteration_result.ref_normal_n0, true);

    iteration_result.angle_model =
        fitRationalAngleModel(planes, iteration_thetas, ref_psi);

    if (!iteration_result.angle_model.valid) {
      throw std::runtime_error("Failed to fit phase-to-angle model.");
    }

    double iteration_max_angle_error = 0.0;
    const double iteration_angle_rmse = computeAngleModelRmse(
        planes, iteration_thetas, iteration_result.angle_model,
        &iteration_max_angle_error);

    std::cout << "  resampling iteration " << iteration + 1
              << ": valid planes=" << valid_count << "/" << final_plane_count
              << ", angle RMSE=" << iteration_angle_rmse * 1000.0 << " mrad";

    if (!previous_phases.empty()) {
      std::cout << ", max phase update=" << max_phase_update << " rad";
    }
    std::cout << std::endl;

    discrete_planes = std::move(planes);
    final_observations = std::move(observations);
    thetas = std::move(iteration_thetas);
    result = iteration_result;

    const bool converged =
        !previous_phases.empty() &&
        max_phase_update <= config.options.angle_resampling_tolerance;

    previous_phases = std::move(refined_phases);
    sampling_model = result.angle_model;

    if (converged) {
      std::cout << "  optical-angle sampling converged." << std::endl;
      break;
    }
  }

  printPlaneQualityDiagnostics(discrete_planes, config.options);

  double max_angle_error = 0.0;
  const double angle_rmse = computeAngleModelRmse(
      discrete_planes, thetas, result.angle_model, &max_angle_error);

  const double theta_min = result.angle_model.evaluate(eff_min_psi);
  const double theta_max = result.angle_model.evaluate(eff_max_psi);

  double min_denominator = std::numeric_limits<double>::infinity();
  double max_denominator = -std::numeric_limits<double>::infinity();

  for (const auto& plane : discrete_planes) {
    if (!plane.valid) {
      continue;
    }
    const double delta_psi = plane.psi - result.angle_model.psi_ref;
    const double denominator =
        result.angle_model.b0 + result.angle_model.b1 * delta_psi;
    min_denominator = std::min(min_denominator, denominator);
    max_denominator = std::max(max_denominator, denominator);
  }

  std::cout << "\nAngle model diagnostics" << std::endl;
  std::cout << "  optical angle range: [" << theta_min << ", " << theta_max
            << "] rad" << std::endl;
  std::cout << "  weighted RMSE: " << angle_rmse * 1000.0 << " mrad"
            << std::endl;
  std::cout << "  maximum absolute angle residual: " << max_angle_error * 1000.0
            << " mrad" << std::endl;
  std::cout << "  denominator range: [" << min_denominator << ", "
            << max_denominator << "]" << std::endl;
  printNormalAxisDiagnostics(discrete_planes, result.nominal_axis_w);

  printPosePlaneResidualDiagnostics(
      "Observed discrete-plane residuals by training pose", discrete_planes,
      final_observations, config.train_poses);

  std::vector<DiscretePlane> consistent_planes;
  consistent_planes.reserve(discrete_planes.size());

  std::vector<double> model_thetas(discrete_planes.size(), 0.0);

  for (std::size_t i = 0; i < discrete_planes.size(); ++i) {
    const auto& source_plane = discrete_planes[i];

    if (!source_plane.valid) {
      consistent_planes.push_back(source_plane);
      continue;
    }

    const double theta = result.angle_model.evaluate(source_plane.psi);
    model_thetas[i] = theta;

    cv::Vec3d model_normal =
        rotateAroundAxis(result.ref_normal_n0, result.nominal_axis_w, theta);
    model_normal =
        cv::normalize(model_normal - model_normal.dot(result.nominal_axis_w) *
                                         result.nominal_axis_w);

    auto plane =
        fitOffsetForFixedNormal(final_observations[i], source_plane.psi,
                                model_normal, source_plane, config.options);

    consistent_planes.push_back(plane);
  }

  normalizePlaneConfidences(consistent_planes);

  printPlaneQualityDiagnostics(consistent_planes, config.options);
  printPosePlaneResidualDiagnostics(
      "Continuous-normal plane residuals by training pose", consistent_planes,
      final_observations, config.train_poses);

  cv::Vec3d consistent_center;
  if (!solveCenterFromPlanes(consistent_planes, result.basis_u, result.basis_v,
                             consistent_center)) {
    throw std::runtime_error("Failed to solve model-consistent scan center.");
  }
  result.nominal_center_s0 = consistent_center;

  cv::Vec3d reference_center;
  result.harmonic_drift = fitHarmonicDriftModel(
      consistent_planes, model_thetas, result.nominal_axis_w,
      result.nominal_center_s0, result.basis_u, result.basis_v,
      reference_center, config.options.harmonic_order,
      config.options.harmonic_regularization);
  result.nominal_center_s0 = reference_center;

  double harmonic_max_offset_mm = 0.0;
  const double harmonic_offset_rmse_mm = computeHarmonicPlaneOffsetRmse(
      result, consistent_planes, &harmonic_max_offset_mm);

  std::cout << "\nContinuous plane model diagnostics" << std::endl;
  std::cout << "  harmonic plane-offset RMSE: " << std::fixed
            << std::setprecision(5) << harmonic_offset_rmse_mm << " mm"
            << std::endl;
  std::cout << "  maximum absolute plane-offset residual: "
            << harmonic_max_offset_mm << " mm" << std::endl;

  double sum_train_pose_sq = 0.0;
  int train_eval_count = 0;

  std::cout << "\n[Step 5] Reconstruction diagnostics on training poses:"
            << std::endl;

  for (std::size_t i = 0; i < train_count; ++i) {
    const int pid = config.train_poses[i];

    const auto diagnostics = evaluateReconstructionDetailed(
        result, train_phase_maps[i], camera_calib.camera_matrix,
        camera_calib.distortion_coefficients, train_board_planes[i],
        eff_min_psi, eff_max_psi, 8);

    const double pose_error = diagnostics.rmse_3d_mm;

    if (pose_error > 0.0 && std::isfinite(pose_error)) {
      sum_train_pose_sq += pose_error * pose_error;
      ++train_eval_count;

      std::cout << "  train pose " << std::setw(2) << pid
                << ": 3D RMSE=" << std::fixed << std::setprecision(5)
                << pose_error << " mm"
                << ", model-plane RMSE=" << diagnostics.model_plane_rmse_mm
                << " mm"
                << ", |n.r| median=" << diagnostics.ray_plane_denom_median
                << ", p05=" << diagnostics.ray_plane_denom_p05
                << ", amplification=" << diagnostics.amplification << std::endl;
    }
  }

  result.train_rmse_mm =
      (train_eval_count > 0)
          ? std::sqrt(sum_train_pose_sq / static_cast<double>(train_eval_count))
          : 0.0;

  double sum_test_pose_sq = 0.0;
  int test_eval_count = 0;

  if (!config.test_poses.empty()) {
    std::cout << "\n[Step 6] Reconstruction diagnostics on test poses:"
              << std::endl;
  }

  for (const int test_pid : config.test_poses) {
    if (test_pid < 0 || test_pid >= static_cast<int>(all_phase_maps.size()) ||
        test_pid >= static_cast<int>(camera_calib.rotation_vectors.size()) ||
        test_pid >= static_cast<int>(camera_calib.translation_vectors.size()) ||
        all_phase_maps[static_cast<std::size_t>(test_pid)].empty()) {
      std::cout << "  test pose " << test_pid
                << ": skipped (invalid or missing data)" << std::endl;
      continue;
    }

    const auto test_board_plane =
        computeBoardPlane(camera_calib.rotation_vectors[test_pid],
                          camera_calib.translation_vectors[test_pid]);

    const auto diagnostics = evaluateReconstructionDetailed(
        result, all_phase_maps[static_cast<std::size_t>(test_pid)],
        camera_calib.camera_matrix, camera_calib.distortion_coefficients,
        test_board_plane, eff_min_psi, eff_max_psi, 4,
        config.options.diagnostic_phase_bins);

    const double pose_error = diagnostics.rmse_3d_mm;

    if (pose_error > 0.0 && std::isfinite(pose_error)) {
      sum_test_pose_sq += pose_error * pose_error;
      ++test_eval_count;

      std::cout << "  test pose " << std::setw(2) << test_pid
                << ": 3D RMSE=" << std::fixed << std::setprecision(5)
                << pose_error << " mm"
                << ", model-plane RMSE=" << diagnostics.model_plane_rmse_mm
                << " mm"
                << ", |n.r| median=" << diagnostics.ray_plane_denom_median
                << ", p05=" << diagnostics.ray_plane_denom_p05
                << ", amplification=" << diagnostics.amplification << std::endl;
      printPhaseBinDiagnostics(diagnostics, eff_min_psi, eff_max_psi);
    }
  }

  result.test_rmse_mm =
      (test_eval_count > 0)
          ? std::sqrt(sum_test_pose_sq / static_cast<double>(test_eval_count))
          : 0.0;

  std::cout << "\n================ Calibration Summary ================"
            << std::endl;
  std::cout << "Nominal Axis (w): [" << result.nominal_axis_w[0] << ", "
            << result.nominal_axis_w[1] << ", " << result.nominal_axis_w[2]
            << "]" << std::endl;
  std::cout << "Nominal Center (S0): [" << result.nominal_center_s0[0] << ", "
            << result.nominal_center_s0[1] << ", "
            << result.nominal_center_s0[2] << "] mm" << std::endl;
  std::cout << "Angle Model: rational trend" << std::endl;
  std::cout << "  tan(theta_base) = dpsi / (" << result.angle_model.b0 << " + "
            << result.angle_model.b1 << "*dpsi), dpsi = psi - "
            << result.angle_model.psi_ref << std::endl;
  std::cout << "Harmonic Drift Order: " << result.harmonic_drift.order
            << " (valid=" << (result.harmonic_drift.valid ? "YES" : "NO") << ")"
            << std::endl;

  if (result.harmonic_drift.valid && result.harmonic_drift.order > 0) {
    std::cout << "Harmonic basis is anchored at the reference angle "
              << "(delta S(0) = 0)." << std::endl;
  }

  std::cout << "Closed-Loop TRAIN 3D RMSE: " << result.train_rmse_mm << " mm"
            << std::endl;
  std::cout << "Closed-Loop TEST 3D RMSE:  " << result.test_rmse_mm << " mm"
            << std::endl;
  std::cout << "=====================================================\n"
            << std::endl;

  return result;
}

}  // namespace msm3d

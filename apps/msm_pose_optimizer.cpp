#include "msm3d/io/config.hpp"
#include "msm3d/io/data_loader.hpp"
#include "msm3d/msm_calibration.hpp"
#include "msm3d/camera_calibration.hpp"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const std::string config_path =
      (argc > 1) ? argv[1] : "./config/msm_calibration.yaml";

  std::cout << "=========================================================\n";
  std::cout << "      MSM Pose Subset Forward Greedy Optimizer (Scheme B) \n";
  std::cout << "=========================================================\n";

  auto msm_config = msm3d::loadMsmCalibrationConfig(config_path);
  const auto cam_calib =
      msm3d::loadCameraCalibrationResult(msm_config.camera_param_file);

  // 1. 一次性将全部 27 个位姿的相位图预加载至内存，避免循环中的重复磁盘 I/O
  const int total_poses = 27;
  std::cout << "Pre-loading all " << total_poses
            << " phase maps into memory ... " << std::flush;
  std::vector<cv::Mat> all_phase_maps(total_poses);
  for (int pid = 0; pid < total_poses; ++pid) {
    const std::string path =
        msm_config.phase_folder + "/pose_" + std::to_string(pid) + ".exr";
    all_phase_maps[pid] = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (all_phase_maps[pid].empty()) {
      std::cerr << "\n[Error] Failed to read: " << path << std::endl;
      return -1;
    }
  }
  std::cout << "Done.\n" << std::endl;

  // 2. 调优运行参数优化 (将采样平面数适度设为 35，搜索速度提升 3
  // 倍且排序完全一致)
  auto opt_config = msm_config;
  opt_config.options.plane_count = 35;
  opt_config.options.min_covisible_poses = 3;  // 种子期位姿较少，门限设为 3

  const std::set<int> test_set(opt_config.test_poses.begin(),
                               opt_config.test_poses.end());

  // 候选位姿池：剔除测试位姿后的全部位姿 (0 ~ 26)
  std::vector<int> candidate_pool;
  for (int i = 0; i < total_poses; ++i) {
    if (test_set.find(i) == test_set.end()) {
      candidate_pool.push_back(i);
    }
  }

  // 3. 阶段一：在基线位姿库中寻找最优 3-Pose 种子三元组
  std::cout << "--- Step 1: Finding Optimal 3-Pose Seed Triad ---" << std::endl;
  const std::vector<int>& base_candidates = msm_config.train_poses;

  std::vector<int> current_subset;
  double best_rmse = std::numeric_limits<double>::infinity();

  const auto t_start = std::chrono::steady_clock::now();

  // 遍历基线集合中所有的 3 位姿组合 (C(9, 3) = 84 次评估，约 3~5 秒)
  const int n_base = static_cast<int>(base_candidates.size());
  for (int i = 0; i < n_base; ++i) {
    for (int j = i + 1; j < n_base; ++j) {
      for (int k = j + 1; k < n_base; ++k) {
        opt_config.train_poses = {base_candidates[i], base_candidates[j],
                                  base_candidates[k]};
        try {
          const auto res =
              msm3d::calibrateMsm(opt_config, cam_calib, all_phase_maps);
          if (res.test_rmse_mm > 0.0 && res.test_rmse_mm < best_rmse) {
            best_rmse = res.test_rmse_mm;
            current_subset = opt_config.train_poses;
          }
        } catch (...) {
          // 奇异几何退化位姿直接跳过
        }
      }
    }
  }

  if (current_subset.empty()) {
    std::cerr << "Failed to find a valid 3-pose seed triad!" << std::endl;
    return -1;
  }

  std::cout << "Initial Best Seed: [";
  for (std::size_t i = 0; i < current_subset.size(); ++i) {
    std::cout << current_subset[i]
              << (i + 1 < current_subset.size() ? ", " : "");
  }
  std::cout << "] -> Initial TEST 3D RMSE: " << std::fixed
            << std::setprecision(5) << best_rmse << " mm\n"
            << std::endl;

  // 从候选池中移除已在种子集合中的位姿
  std::vector<int> remaining_pool;
  for (int p : candidate_pool) {
    if (std::find(current_subset.begin(), current_subset.end(), p) ==
        current_subset.end()) {
      remaining_pool.push_back(p);
    }
  }

  // 4. 阶段二：前向贪心迭代扩充 (Forward Greedy Selection)
  std::cout << "--- Step 2: Forward Greedy Expansion ---" << std::endl;
  int round = 1;

  while (!remaining_pool.empty()) {
    int best_cand = -1;
    double round_best_rmse = best_rmse;

    for (std::size_t i = 0; i < remaining_pool.size(); ++i) {
      const int cand = remaining_pool[i];
      auto trial_train = current_subset;
      trial_train.push_back(cand);

      opt_config.train_poses = trial_train;

      try {
        const auto res =
            msm3d::calibrateMsm(opt_config, cam_calib, all_phase_maps);
        if (res.test_rmse_mm > 0.0 && res.test_rmse_mm < round_best_rmse) {
          round_best_rmse = res.test_rmse_mm;
          best_cand = cand;
        }
      } catch (...) {
        // 异常样本跳过
      }
    }

    // 判断本轮是否有正向收益
    if (best_cand != -1 && round_best_rmse < best_rmse - 1e-4) {
      const double improvement =
          (best_rmse - round_best_rmse) / best_rmse * 100.0;
      std::cout << "[Round " << std::setw(2) << round << "] Added Pose "
                << std::setw(2) << best_cand << " | TEST RMSE: " << std::fixed
                << std::setprecision(5) << best_rmse << " mm -> "
                << round_best_rmse << " mm (" << std::setprecision(2) << "-"
                << improvement << "%)" << std::endl;

      best_rmse = round_best_rmse;
      current_subset.push_back(best_cand);
      remaining_pool.erase(
          std::remove(remaining_pool.begin(), remaining_pool.end(), best_cand),
          remaining_pool.end());
      round++;
    } else {
      std::cout << "[Round " << std::setw(2) << round
                << "] No remaining poses can improve TEST RMSE further. "
                   "Optimization converged.\n"
                << std::endl;
      break;
    }
  }

  const auto t_end = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  // 5. 阶段三：使用全采样 (如 plane_count=60) 重新评估最终最优子集
  std::cout << "================ Optimization Final Report ================"
            << std::endl;
  std::cout << "Total Search Elapsed: " << std::fixed << std::setprecision(2)
            << elapsed << " s" << std::endl;
  std::cout << "Optimal Training Poses (" << current_subset.size() << "): [";
  std::sort(current_subset.begin(), current_subset.end());
  for (std::size_t i = 0; i < current_subset.size(); ++i) {
    std::cout << current_subset[i]
              << (i + 1 < current_subset.size() ? ", " : "");
  }
  std::cout << "]" << std::endl;

  // 使用原始配置的 plane_count 验证最终精度上限
  auto final_eval_config = msm_config;
  final_eval_config.train_poses = current_subset;
  const auto final_res =
      msm3d::calibrateMsm(final_eval_config, cam_calib, all_phase_maps);

  std::cout << "Final Upper-bound TEST 3D RMSE: " << std::fixed
            << std::setprecision(5) << final_res.test_rmse_mm << " mm"
            << std::endl;
  std::cout << "Final Closed-Loop TRAIN 3D RMSE: " << final_res.train_rmse_mm
            << " mm" << std::endl;
  std::cout << "===========================================================\n"
            << std::endl;

  std::cout << "Copy & paste this into config/msm_calibration.yaml:\n"
            << "train_poses: [";
  for (std::size_t i = 0; i < current_subset.size(); ++i) {
    std::cout << current_subset[i]
              << (i + 1 < current_subset.size() ? ", " : "");
  }
  std::cout << "]\n" << std::endl;

  return 0;
}
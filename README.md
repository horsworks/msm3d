# MSM3D

C++ calibration and 3D reconstruction pipeline for a MEMS scanning-mirror structured-light system.

## Current pipeline

1. **Camera calibration**
   - circle-grid / chessboard detection
   - OpenCV intrinsic/extrinsic calibration
   - reprojection statistics and optional residual diagnostics
2. **Phase processing**
   - multi-step wrapped phase estimation
   - three-frequency absolute phase unwrapping
   - phase-quality rejection and mask-aware median filtering
3. **Scanning-mirror calibration**
   - subpixel iso-phase extraction
   - pose-balanced robust light-plane fitting
   - fixed scan-axis geometry estimation
   - centered rational phase-to-angle model
   - optical-angle-uniform resampling
   - anchored second-order harmonic equivalent-axis drift model
   - closed-loop 3D reconstruction evaluation

The stable pre-Ceres baseline uses train pose IDs `[7, 8, 10, 11, 16]` and test pose IDs `[2, 6]`.

- Closed-loop TRAIN 3D RMSE: **0.07410 mm**
- Closed-loop TEST 3D RMSE: **0.06870 mm**

## Project layout

```text
include/msm3d/
  core/      common geometry types
  camera/    camera calibration, diagnostics, serialization
  phase/     phase processing and quality definitions
  msm/       scanning-mirror model, calibration, reconstruction, serialization
  io/        YAML configuration and dataset loading

src/
  core/
  camera/
  phase/
  msm/
  io/

apps/
  camera/
  phase/
  msm/
```

## Build

Dependencies:

- C++17
- CMake 3.28+
- Eigen3
- OpenCV
- yaml-cpp

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Main executables

```bash
build/cam_demo
build/phase_demo
build/msm_demo
```

## Status

Under active development.

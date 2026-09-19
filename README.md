# MSM3D

C++ implementation of a MEMS scanning-mirror structured-light calibration and
3D reconstruction pipeline.

The current project focuses on a continuous light-plane model for a resonant
MEMS scanning mirror. Camera calibration and phase processing provide the
reference geometry and absolute phase observations; MSM calibration estimates
a continuous light plane as a function of phase.

## Current pipeline

1. **Camera calibration**
   - Circle-grid detection with OpenCV.
   - Radial/tangential distortion calibration.
   - Reprojection error and per-view residual diagnostics.

2. **Phase processing**
   - 24-step sinusoidal phase fitting for frequencies `[60, 55, 51]`.
   - Three-frequency absolute phase unwrapping.
   - Modulation, sinusoidal-fit, saturation, and frequency-consistency quality checks.
   - Mask-aware `3x3` median filtering for the stable configuration.

3. **MEMS scanning-mirror calibration**
   - Multi-pose iso-phase subpixel extraction using local linear phase fitting.
   - Pose-balanced robust TLS fitting of discrete light planes.
   - Fixed-axis continuous normal model.
   - Centered rational phase-to-optical-angle mapping.
   - Uniform optical-angle resampling.
   - Anchored second-order harmonic equivalent-axis drift model.
   - Train/test reconstruction and phase-bin error diagnostics.

The continuous plane model is evaluated as

```text
alpha = atan2(psi - psi_ref, b0 + b1 * (psi - psi_ref))
n(alpha) = R(w, alpha) * n0
S(alpha) = S_ref + u * delta_u(alpha) + v * delta_v(alpha)
d(alpha) = -n(alpha)^T * S(alpha)
```

The harmonic drift basis is anchored at the reference angle, so
`delta S(0) = 0`.

## Validated baseline

Current stable MSM configuration:

```text
train pose IDs: [7, 8, 10, 11, 16]
test pose IDs:  [2, 6]
phase filter:   3x3 median
harmonic order: 2
```

Validated closed-loop reconstruction result:

```text
TRAIN 3D RMSE: 0.07410 mm
TEST  3D RMSE: 0.06870 mm
```

Several experimental branches that did not improve held-out 3D accuracy have
been removed from the production path, including multi-frequency phase fusion,
local-plane phase filtering, residual angle correction, camera-center
refinement, point exclusion, and dense linear MSM refinement.

## Build

Dependencies:

- C++17
- CMake / Ninja
- OpenCV
- Eigen3
- yaml-cpp

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
build/cam_demo
build/phase_demo
build/msm_demo
```

Configuration files are under `config/`. Runtime data and generated outputs are
kept under `data/` and `output/` and are ignored by Git.

## Next step

The next development milestone is Ceres-based nonlinear optimization of the
continuous MSM model, followed by joint optimization with board poses if the
MSM-only optimization provides a stable improvement.

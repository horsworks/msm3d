# MSM3D: Harmonic Light-Plane Calibration for MEMS Scanning-Mirror 3D Reconstruction

MSM3D is a C++ project for calibration and 3D reconstruction
using MEMS scanning-mirror structured light.

## Status

Under active development.

## Development Environment

- Ubuntu 24.04 / WSL2
- C++20
- CMake
- Ninja
- GCC

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
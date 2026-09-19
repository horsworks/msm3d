add_library(msm3d_core
  core/geometry.cpp
  camera/calibration.cpp
  camera/diagnostics.cpp
  camera/serialization.cpp
  phase/processor.cpp
  io/config.cpp
  io/data_loader.cpp
  msm/model.cpp
  msm/iso_phase.cpp
  msm/plane_fitting.cpp
  msm/geometry_solver.cpp
  msm/reconstruction.cpp
  msm/diagnostics.cpp
  msm/calibration.cpp
  msm/serialization.cpp
)

add_library(msm3d::core ALIAS msm3d_core)

target_include_directories(msm3d_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(msm3d_core
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
  PRIVATE
    yaml-cpp
)

target_compile_features(msm3d_core PUBLIC cxx_std_17)

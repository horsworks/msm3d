cmake_minimum_required(VERSION 3.28)

project(msm3d VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
    include
)

add_library(msm3d
    src/camera_calibration.cpp
    src/phase_processor.cpp
    src/geometry.cpp
    src/io/config.cpp
    src/io/data_loader.cpp
    src/msm_calibration.cpp
)

target_link_libraries(msm3d
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)

add_executable(cam_demo
    apps/camera_calibration_demo.cpp
)

target_link_libraries(cam_demo
    PRIVATE
    msm3d
)

add_executable(phase_demo
    apps/phase_processor_demo.cpp
)

target_link_libraries(phase_demo
    PRIVATE
    msm3d
)

add_executable(iso_demo
    apps/inspect_iso_phase.cpp
)

target_link_libraries(iso_demo
    PRIVATE
    msm3d
)
add_executable(phase_map_demo
    apps/inspect_phase_map.cpp
)

target_link_libraries(phase_map_demo
    PRIVATE
    msm3d
)

add_executable(msm_demo
    apps/msm_calibration_demo.cpp
)

target_link_libraries(msm_demo
    PRIVATE
    msm3d
)

add_executable(pose_optimizer 
    apps/msm_pose_optimizer.cpp
)

target_link_libraries(pose_optimizer
    PRIVATE
    msm3d
)

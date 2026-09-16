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
    src/phase.cpp
    src/dataset.cpp
    src/config.cpp
    src/geometry.cpp
)

target_link_libraries(msm3d
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)

add_executable(camera_calibration_demo
    apps/camera_calibration_demo.cpp
)

target_link_libraries(camera_calibration_demo
    msm3d
)

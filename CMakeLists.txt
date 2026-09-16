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
)

target_link_libraries(msm3d
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)

add_executable(demo
    apps/camera_calibration_demo.cpp
)

target_link_libraries(demo
    msm3d
)

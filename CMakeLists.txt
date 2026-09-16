cmake_minimum_required(VERSION 3.28)

project(msm3d VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(msm3d
    src/geometry.cpp
    src/camera.cpp
    src/phase.cpp
    src/dataset.cpp
)

target_include_directories(msm3d PUBLIC include)

target_link_libraries(msm3d
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

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

add_executable(
    phase_demo
    apps/phase_demo.cpp
)

target_link_libraries(
    phase_demo
    PRIVATE
    msm3d
)

enable_testing()

add_executable(
    test_phase
    tests/test_phase.cpp
)

target_link_libraries(
    test_phase
    PRIVATE
    msm3d
)

add_test(
    NAME phase_test
    COMMAND test_phase
)
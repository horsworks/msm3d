cmake_minimum_required(VERSION 3.28)

project(
    msm3d
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

add_library(
    msm3d
    src/geometry.cpp
    src/camera.cpp
    src/phase.cpp
)

target_include_directories(
    msm3d
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    msm3d
    PUBLIC
        Eigen3::Eigen
)

# tests
enable_testing()

add_executable(
    test_geometry

    tests/test_geometry.cpp
)

target_link_libraries(
    test_geometry

    PRIVATE
        msm3d
)

add_test(
    NAME geometry_test

    COMMAND test_geometry
)
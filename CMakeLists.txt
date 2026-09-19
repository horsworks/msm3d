cmake_minimum_required(VERSION 3.28)

project(msm3d VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MSM3D_BUILD_APPS "Build MSM3D command-line applications" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(src)

if(MSM3D_BUILD_APPS)
  add_subdirectory(apps)
endif()

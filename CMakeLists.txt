cmake_minimum_required(VERSION 3.28)

project(
    msm3d
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

add_executable(
    msm3d_smoke_test
    tests/smoke_test.cpp
)

target_compile_features(
    msm3d_smoke_test
    PRIVATE
        cxx_std_17
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        msm3d_smoke_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

add_test(
    NAME msm3d.smoke
    COMMAND msm3d_smoke_test
)
cmake_minimum_required(VERSION 3.16)
project(insight_vio_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(INSIGHT_VIO_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(INSIGHT_VIO_BUILD_BENCH "Build micro-benchmarks" ON)

include(FetchContent)

# Header-only Eigen: fetch the tarball and use includes only (skip Eigen's CMake).
# SOURCE_SUBDIR points at a non-existent path so FetchContent_MakeAvailable does not
# add Eigen's project (incompatible with CMake 4.x check macros).
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR unavailable
)
FetchContent_MakeAvailable(eigen)
add_library(insight_vio_eigen INTERFACE)
target_include_directories(insight_vio_eigen INTERFACE ${eigen_SOURCE_DIR})

add_library(insight_vio_cpp
  src/preintegration.cpp
  src/triangulation.cpp
)
add_library(insight_vio::cpp ALIAS insight_vio_cpp)
target_include_directories(insight_vio_cpp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(insight_vio_cpp PUBLIC insight_vio_eigen)
target_compile_features(insight_vio_cpp PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(insight_vio_cpp PRIVATE /W4)
else()
  target_compile_options(insight_vio_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(INSIGHT_VIO_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  set(INSIGHT_VIO_GOLDENS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens")

  add_executable(test_lie tests/test_lie.cpp)
  target_link_libraries(test_lie PRIVATE insight_vio_cpp GTest::gtest_main)
  add_test(NAME test_lie COMMAND test_lie)

  add_executable(test_preintegration tests/test_preintegration.cpp)
  target_link_libraries(test_preintegration PRIVATE insight_vio_cpp GTest::gtest_main)
  target_compile_definitions(test_preintegration PRIVATE
    INSIGHT_VIO_GOLDENS_DIR="${INSIGHT_VIO_GOLDENS_DIR}")
  add_test(NAME test_preintegration COMMAND test_preintegration)

  add_executable(test_triangulation tests/test_triangulation.cpp)
  target_link_libraries(test_triangulation PRIVATE insight_vio_cpp GTest::gtest_main)
  add_test(NAME test_triangulation COMMAND test_triangulation)
endif()

if(INSIGHT_VIO_BUILD_BENCH)
  add_executable(insight_vio_microbench bench/microbench.cpp)
  target_link_libraries(insight_vio_microbench PRIVATE insight_vio_cpp)
endif()

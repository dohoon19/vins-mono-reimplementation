
cmake_minimum_required(VERSION 3.5)
project(image_feature_extractor)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(algorithm)

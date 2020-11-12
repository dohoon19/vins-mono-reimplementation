
cmake_minimum_required(VERSION 3.5)
project(visual_inertial_slam)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(algorithm)

add_subdirectory(algorithm)
add_subdirectory(ros2_msgs)
add_subdirectory(ros2)

ament_package()

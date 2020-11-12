cmake_minimum_required(VERSION 3.5)
project(pose_estimator_ros2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Ceres REQUIRED)

add_executable(
  ${PROJECT_NAME} 
  ${PROJECT_NAME}.cpp
)

ament_target_dependencies(
  ${PROJECT_NAME}  
  rclcpp
  geometry_msgs
  nav_msgs
)

target_link_libraries(
  ${PROJECT_NAME}
  pose_estimator
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include
)

rosidl_target_interfaces(
  ${PROJECT_NAME}
  image_feature_extractor_ros2_msgs "rosidl_typesupport_cpp"
)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

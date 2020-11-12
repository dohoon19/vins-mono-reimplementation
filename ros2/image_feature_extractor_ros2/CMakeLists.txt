cmake_minimum_required(VERSION 3.5)
project(image_feature_extractor_ros2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(
  ${PROJECT_NAME} 
  ${PROJECT_NAME}.cpp
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
)

target_link_libraries(
  ${PROJECT_NAME}
  image_feature_extractor
  ${OpenCV_LIBS}
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include
  ${OpenCV_INCLUDE_DIR}
)

rosidl_target_interfaces(
  ${PROJECT_NAME}
  ${PROJECT_NAME}_msgs "rosidl_typesupport_cpp"
)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)
install(
  FILES parameter.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

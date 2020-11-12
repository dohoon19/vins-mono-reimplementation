
cmake_minimum_required(VERSION 3.5)
project(image_feature_extractor)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(
  ${PROJECT_NAME} STATIC
  ${PROJECT_NAME}.cpp
  camera.cpp
)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}
)
install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(
  ${PROJECT_NAME}_tester
  ${PROJECT_NAME}_tester.cpp
)
target_link_libraries(
  ${PROJECT_NAME}_tester PUBLIC
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

install(
  TARGETS ${PROJECT_NAME}_tester
  DESTINATION lib/${PROJECT_NAME}_tester
)
    

cmake_minimum_required(VERSION 3.5)
project(image_feature_extractor)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(
    ${PROJECT_NAME} STATIC
    image_feature_extractor.cpp
    camera.cpp
)
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME})

add_executable(
    ${PROJECT_NAME}_tester
    image_feature_extractor_tester.cpp
)
target_link_libraries(
    ${PROJECT_NAME}_tester PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME}
)
    
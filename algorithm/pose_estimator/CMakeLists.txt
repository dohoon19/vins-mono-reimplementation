
cmake_minimum_required(VERSION 3.5)
project(pose_estimator)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(
    ${PROJECT_NAME} STATIC
    ${PROJECT_NAME}.cpp
)
target_link_libraries(
    ${PROJECT_NAME} PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${CERES_LIBRARIES}
)
install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib/${PROJECT_NAME})

# add_executable(
#     ${PROJECT_NAME}_tester
#     ${PROJECT_NAME}_tester.cpp
# )
# target_link_libraries(
#     ${PROJECT_NAME}_tester PUBLIC
#     ${PROJECT_NAME}
#     ${OpenCV_LIBS}
#     Eigen3::Eigen
# )

# install(
#     TARGETS ${PROJECT_NAME}_tester
#     DESTINATION lib/${PROJECT_NAME}_tester
# )
    
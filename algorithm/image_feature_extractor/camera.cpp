#include "camera.hpp"

namespace visual_inertial_slam {
namespace camera {

cv::Point2f Camera::Undistort(const cv::Point2f& image_point) {
  const auto distored_point = Normalize(image_point);

  auto undistorted_point = distored_point;
  for (int i = 0; i < parameters_.iteration_threshold; ++i) {
    const auto error = Distort(undistorted_point) - distored_point;
    undistorted_point = undistorted_point - error;
    if (error.x < parameters_.error_threshold &&
        error.y < parameters_.error_threshold)
      break;
  }

  return Denormalize(undistorted_point);
}

cv::Point2f Camera::Normalize(const cv::Point2f& image_point) {
  cv::Point2f normalized_point;
  normalized_point.y = (image_point.y - parameters_.cy) / parameters_.fy;
  normalized_point.x = (image_point.x - parameters_.cx) / parameters_.fx -
                       parameters_.skew_c * normalized_point.y;
  return normalized_point;
}

cv::Point2f Camera::Denormalize(const cv::Point2f& normalized_point) {
  cv::Point2f denormalized_point;
  denormalized_point.x =
      parameters_.fx *
          (normalized_point.x + parameters_.skew_c * normalized_point.y) +
      parameters_.cx;
  denormalized_point.y = parameters_.fy * normalized_point.y + parameters_.cy;
  return denormalized_point;
}

cv::Point2f Camera::Distort(const cv::Point2f& normalized_point) {
  const auto& param = parameters_;
  const auto squared_radius =
      std::pow(normalized_point.x, 2) + std::pow(normalized_point.y, 2);
  const auto radial_distance =
      1 + param.k1 * param.k2 + param.k2 * squared_radius * squared_radius +
      param.k3 * squared_radius * squared_radius * squared_radius;
  cv::Point2f distorted_point;
  distorted_point.x =
      radial_distance * normalized_point.x +
      2 * param.p1 * normalized_point.x * normalized_point.y +
      param.p2 * (squared_radius + 2 * normalized_point.x * normalized_point.x);
  distorted_point.y = radial_distance * normalized_point.y +
                      param.p1 * (squared_radius +
                                  2 * normalized_point.y * normalized_point.y) +
                      2 * param.p2 * normalized_point.x * normalized_point.y;
  return distorted_point;
}

}  // namespace camera
}  // namespace visual_inertial_slam
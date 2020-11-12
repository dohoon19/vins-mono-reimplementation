#include "reprojection_error.hpp"

namespace visual_inertial_slam {
namespace pose_estimator {
namespace reprojection_error {

ReprojectionError::ReprojectionError(
    const Eigen::Vector3d& feature_at_camera_i,
    const Eigen::Vector3d& feature_at_camera_j) {}
bool ReprojectionError::Evaluate(double const* const* parameters,
                                 double* residuals, double** jacobians) const {
  Eigen::Vector3d camera_i_position = {parameters[0][0], parameters[0][1],
                                       parameters[0][2]};
  Eigen::Quaterniond camera_i_orientation = {
      parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

  Eigen::Vector3d camera_j_position = {parameters[1][0], parameters[1][1],
                                       parameters[1][2]};
  Eigen::Quaterniond camera_j_orientation = {
      parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]};

  double camera_i_feature_depth = parameters[3][0];
  double camera_j_feature_depth = parameters[3][1];

  const auto feature_at_camera_i =
      feature_at_noramlized_camera_i_ * camera_i_feature_depth;
  const auto feature_at_camera_j =
      feature_at_normalized_camera_j_ * camera_j_feature_depth;

  const auto feature_i_at_world_frame =
      camera_i_orientation * feature_at_camera_i + camera_i_position;
  const auto reprojected_feature_at_camera_j =
      camera_j_orientation.inverse() *
      (feature_i_at_world_frame - camera_j_position) * camera_j_feature_depth;

  Eigen::Map<Eigen::Vector2d> residual(residuals);

  residual = reprojected_feature_at_camera_j - feature_at_camera_j;

  Eigen::Vector3d camera_position;
}
}  // namespace reprojection_error
}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#ifndef __REPROJECTION_ERROR_HPP__
#define __REPROJECTION_ERROR_HPP__

#include "Eigen/Dense"
#include "ceres/ceres.h"

namespace visual_inertial_slam {
namespace pose_estimator {
namespace reprojection_error {
class ReprojectionError : public ceres::SizedCostFunction<3, 7, 7, 2> {
 public:
  ReprojectionError(const Eigen::Vector3d& feature_at_noramlized_camera_i,
                    const Eigen::Vector3d& feature_at_normalized_camera_j)
      : feature_at_noramlized_camera_i_(feature_at_noramlized_camera_i),
        feature_at_normalized_camera_j_(feature_at_normalized_camera_j) {}
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const final;

 private:
  Eigen::Vector3d feature_at_noramlized_camera_i_;
  Eigen::Vector3d feature_at_normalized_camera_j_;
}

}  // namespace reprojection_error
}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#endif  // __REPROJECTION_ERROR_HPP__

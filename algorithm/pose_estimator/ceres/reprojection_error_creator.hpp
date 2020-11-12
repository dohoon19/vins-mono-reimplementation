#ifndef __REPROJECTION_ERROR_CREATOR_HPP__
#define __REPROJECTION_ERROR_CREATOR_HPP__

#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace visual_inertial_slam {
namespace pose_estimator {
namespace reprojection_error {

class ReprojectionErrorCreator {
 public:
  ReprojectionErrorCreator(
      const Eigen::Vector3d& feature_at_noramlized_i_camera,
      const Eigen::Vector3d& feature_at_normalized_j_camera)
      : feature_at_noramlized_i_camera_(feature_at_noramlized_i_camera),
        feature_at_normalized_j_camera_(feature_at_normalized_j_camera) {}

  template <typename T>
  bool operator()(const T* const i_camera_translation_in,
                  const T* const i_camera_rotation_in,
                  const T* const j_camera_translation_in,
                  const T* const j_camera_rotation_in,
                  const T* const i_camera_feature_depth, T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> i_camera_translation(
        i_camera_translation_in);
    Eigen::Map<const Eigen::Quaternion<T>> i_camera_rotation(
        i_camera_rotation_in);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> j_camera_translation(
        j_camera_translation_in);
    Eigen::Map<const Eigen::Quaternion<T>> j_camera_rotation(
        j_camera_rotation_in);

    Eigen::Matrix<T, 3, 1> feature_at_i_camera =
        feature_at_noramlized_i_camera_.template cast<T>() *
        i_camera_feature_depth[0];
    Eigen::Matrix<T, 3, 1> feature_i_at_world_frame =
        i_camera_rotation * feature_at_i_camera + i_camera_translation;
    Eigen::Matrix<T, 3, 1> reprojected_feature_at_j_camera =
        j_camera_rotation.inverse() *
        (feature_i_at_world_frame - j_camera_translation);
    reprojected_feature_at_j_camera /= reprojected_feature_at_j_camera[2];
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(residuals);
    residual =
        (reprojected_feature_at_j_camera - feature_at_normalized_j_camera_)
            .head(2);
    // std::cerr << "residual " << residual << std::endl;
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& feature_at_noramlized_i_camera,
      const Eigen::Vector3d& feature_at_noramlized_j_camera) {
    return (
        new ceres::AutoDiffCostFunction<ReprojectionErrorCreator, 2, 3, 4, 3, 4,
                                        1>(new ReprojectionErrorCreator(
            feature_at_noramlized_i_camera, feature_at_noramlized_j_camera)));
  }

 private:
  Eigen::Vector3d feature_at_noramlized_i_camera_;
  Eigen::Vector3d feature_at_normalized_j_camera_;
};

}  // namespace reprojection_error
}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#endif  // __REPROJECTION_ERROR_CREATOR_HPP__

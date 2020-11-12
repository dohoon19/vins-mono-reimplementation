#include "imu_preintegrator.hpp"

namespace visual_inertial_slam {
namespace imu_preintegrator {

ImuPreintegration ImuPreintegrator::GetImuPreintegration(
    const int keyframe_index) {
  return imu_preintegration_queue_[keyframe_index];
}

ImuPreintegration ImuPreintegrator::GetImuPreintegrationMeasurementModel(
    const Eigen::Vector3d& current_position,
    const Eigen::Vector3d& current_rotation,
    const Eigen::Vector3d& current_velocity,
    const Eigen::Vector3d& current_accel_bias,
    const Eigen::Vector3d& current_gyro_bias,
    const Eigen::Vector3d& next_position,
    const Eigen::Vector3d& next_accel_bias,
    const Eigen::Vector3d& next_gyro_bias, const double delta_t) {
  const ImuPreintegration imu_preintegration_measurement_model = {};
  return imu_preintegration_measurement_model;
}

void ImuPreintegrator::Integrate(const int keyframe_index,
                                 const ImuRawData& imu_raw_data) {
  ImuPreintegration imu_preintegration;
  imu_preintegration_queue_[keyframe_index] = imu_preintegration;

  Eigen::MatrixXd imu_kinematics;
  Eigen::MatrixXd noise_kinematics;
  double delta_t;
  Eigen::MatrixXd updated_covariance =
      UpdateCovariance(imu_kinematics, noise_kinematics,
                       covariance_queue_[keyframe_index], delta_t);
  covariance_queue_[keyframe_index] = updated_covariance;

  Eigen::MatrixXd updated_jacobian =
      UpdateJacobian(imu_kinematics, jacobian_queue_[keyframe_index], delta_t);
  jacobian_queue_[keyframe_index] = updated_jacobian;
}

ImuPreintegration ImuPreintegrator::CorrectBias(
    const ImuPreintegration& imu_preintegration, const Eigen::MatrixXd jacobian,
    const Eigen::Vector3d delta_accel_bias,
    const Eigen::Vector3d delta_gyro_bias) {
  ImuPreintegration corrected_imu_integration;
  return corrected_imu_integration;
}

Eigen::MatrixXd ImuPreintegrator::UpdateCovariance(
    const Eigen::MatrixXd& imu_kinematics,
    const Eigen::MatrixXd& noise_kinematics,
    const Eigen::MatrixXd& current_covariance, const double delta_t) {
  Eigen::MatrixXd updated_covariance;
  return updated_covariance;
}

Eigen::MatrixXd ImuPreintegrator::UpdateJacobian(
    const Eigen::MatrixXd& imu_kinematics,
    const Eigen::MatrixXd& current_jacobian, const double delta_t) {
  Eigen::MatrixXd updated_jacobian;
  return updated_jacobian;
}

}  // namespace imu_preintegrator
}  // namespace visual_inertial_slam

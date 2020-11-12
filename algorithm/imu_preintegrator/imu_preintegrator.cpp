#include "imu_preintegrator.hpp"

namespace visual_inertial_slam {
namespace imu_preintegrator {

void ImuPreintegrator::SetImuRawData(const ImuRawData& imu_raw_data) {
  imu_raw_data_buffer_.emplace_back(imu_raw_data);
}

ImuPreintegration ImuPreintegrator::GetImuPreintegration(const int frame_id) {
  return imu_preintegration_queue_[frame_id];
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

void ImuPreintegrator::Preintegrate(const int frame_id,
                                    const ImuRawData& imu_raw_data) {
  ImuPreintegration imu_preintegration;
  imu_preintegration_queue_[frame_id] = imu_preintegration;

  Eigen::MatrixXd imu_kinematics;
  Eigen::MatrixXd noise_kinematics;
  double delta_t;
  Eigen::MatrixXd updated_covariance = UpdateCovariance(
      imu_kinematics, noise_kinematics, covariance_queue_[frame_id], delta_t);
  covariance_queue_[frame_id] = updated_covariance;

  Eigen::MatrixXd updated_jacobian =
      UpdateJacobian(imu_kinematics, jacobian_queue_[frame_id], delta_t);
  jacobian_queue_[frame_id] = updated_jacobian;
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

#include "imu_preintegrator.hpp"

namespace visual_inertial_slam {
namespace imu_preintegrator {

void ImuPreintegrator::SetImuRawData(const ImuRawData& imu_raw_data) {
  if (frame_queue_.size() < 2) return;

  const auto& last_frame = frame_queue_.rbegin()->second;
  auto& last_second_frame = std::next(frame_queue_.rbegin())->second;
  if (last_frame.time > imu_raw_data.time) {
    Preintegrate(last_second_frame.id, imu_raw_data);
    last_second_frame.imu_raw_data_list.emplace_back(imu_raw_data);
  } else {
    imu_raw_data_buffer_.emplace_back(imu_raw_data);
  }
}

bool ImuPreintegrator::SetFrame(const double time, const int frame_id) {
  if (frame_queue_.count(frame_id) > 0) return false;
  const auto& last_frame = frame_queue_.rbegin()->second;
  if (last_frame.time > time) return false;
  if (last_frame.id > frame_id) return false;

  frame_queue_[frame_id].id = frame_id;
  frame_queue_[frame_id].time = time;
  for (auto imu_raw_data = imu_raw_data_buffer_.begin();
       imu_raw_data < imu_raw_data_buffer_.end();) {
    if (last_frame.time < imu_raw_data->time) break;
    Preintegrate(frame_id, *imu_raw_data);
    frame_queue_[frame_id].imu_raw_data_list.emplace_back(*imu_raw_data);
    imu_raw_data = imu_raw_data_buffer_.erase(imu_raw_data);
  }
}

ImuPreintegration ImuPreintegrator::GetImuPreintegration(const int frame_id) {
  return frame_queue_[frame_id].imu_preintegration;
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
  // TODO
  frame_queue_[frame_id].imu_preintegration = imu_preintegration;

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

void Integrate(double dt, const Eigen::Vector3d& accel_i,
               const Eigen::Vector3d& gyro_i, const Eigen::Vector3d& accel_j,
               const Eigen::Vector3d& gyro_j, const Eigen::Vector3d& delta_p,
               const Eigen::Vector3d& delta_v,
               const Eigen::Quaterniond& delta_q,
               const Eigen::Vector3d& linearized_ba,
               const Eigen::Vector3d& linearized_bg,
               Eigen::Vector3d& integrated_delta_p,
               Eigen::Quaterniond& integrated_delta_q,
               Eigen::Vector3d& integrated_delta_v) {
  // mid point integration

  Eigen::Vector3d unbiased_accel_i = delta_q * (accel_i - linearized_ba);
  Eigen::Vector3d unbiased_gyro = 0.5 * (gyro_i + gyro_j) - linearized_bg;
  // 이상함?
  integrated_delta_q =
      delta_q * Eigen::Quaterniond(1, unbiased_gyro(0) * dt / 2,
                                   unbiased_gyro(1) * dt / 2,
                                   unbiased_gyro(2) * dt / 2);
  Eigen::Vector3d unbiased_accel_j =
      integrated_delta_q * (accel_j - linearized_ba);
  Eigen::Vector3d unbiased_accel = 0.5 * (unbiased_accel_i + unbiased_accel_j);
  integrated_delta_p = delta_p + delta_v * dt + 0.5 * unbiased_accel * dt * dt;
  integrated_delta_v = delta_v + unbiased_accel * dt;

  Eigen::Vector3d w_x = 0.5 * (gyro_i + gyro_j) - linearized_bg;
  Eigen::Vector3d a_0_x = accel_i - linearized_ba;
  Eigen::Vector3d a_1_x = accel_j - linearized_ba;
  Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

  R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
  R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
      a_0_x(0), 0;
  R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
      a_1_x(0), 0;

  Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
  F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(0, 3) =
      -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt * dt +
      -0.25 * integrated_delta_q.toRotationMatrix() * R_a_1_x *
          (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
  F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3, 3) * dt;
  F.block<3, 3>(0, 9) =
      -0.25 *
      (delta_q.toRotationMatrix() + integrated_delta_q.toRotationMatrix()) *
      dt * dt;
  F.block<3, 3>(0, 12) =
      -0.25 * integrated_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * -dt;
  F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
  F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3, 3) * dt;
  F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * dt +
                        -0.5 * integrated_delta_q.toRotationMatrix() * R_a_1_x *
                            (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
  F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(6, 9) =
      -0.5 *
      (delta_q.toRotationMatrix() + integrated_delta_q.toRotationMatrix()) * dt;
  F.block<3, 3>(6, 12) =
      -0.5 * integrated_delta_q.toRotationMatrix() * R_a_1_x * dt * -dt;
  F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
  F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
  // cout<<"A"<<endl<<A<<endl;

  Eigen::MatrixXd V = Eigen::Eigen::MatrixXd::Zero(15, 18);
  V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * dt * dt;
  V.block<3, 3>(0, 3) = 0.25 * -integrated_delta_q.toRotationMatrix() *
                        R_a_1_x * dt * dt * 0.5 * dt;
  V.block<3, 3>(0, 6) = 0.25 * integrated_delta_q.toRotationMatrix() * dt * dt;
  V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
  V.block<3, 3>(3, 3) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
  V.block<3, 3>(3, 9) = 0.5 * Eigen::MatrixXd::Identity(3, 3) * dt;
  V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * dt;
  V.block<3, 3>(6, 3) =
      0.5 * -integrated_delta_q.toRotationMatrix() * R_a_1_x * dt * 0.5 * dt;
  V.block<3, 3>(6, 6) = 0.5 * integrated_delta_q.toRotationMatrix() * dt;
  V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
  V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3, 3) * dt;
  V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3, 3) * dt;

  // step_jacobian = F;
  // step_V = V;
  jacobian = F * jacobian;
  covariance = F * covariance * F.transpose() + V * noise * V.transpose();
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

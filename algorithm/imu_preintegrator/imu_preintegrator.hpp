#include <map>
#include <unordered_map>

#include "eigen3/Eigen/Dense"

namespace visual_inertial_slam {
namespace imu_preintegrator {

struct ImuRawData {
  double time;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};

struct ImuPreintegration {
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  Eigen::Quaterniond rotation_q;
  Eigen::Vector3d accel_bias;
  Eigen::Vector3d gyro_bias;
};

class ImuPreintegrator {
 public:
  ImuPreintegration GetImuPreintegration(const int keyframe_index);
  ImuPreintegration GetImuPreintegrationMeasurementModel(
      const Eigen::Vector3d& current_position,
      const Eigen::Vector3d& current_rotation,
      const Eigen::Vector3d& current_velocity,
      const Eigen::Vector3d& current_accel_bias,
      const Eigen::Vector3d& current_gyro_bias,
      const Eigen::Vector3d& next_position,
      const Eigen::Vector3d& next_accel_bias,
      const Eigen::Vector3d& next_gyro_bias, const double delta_t);
  void Integrate(const int keyframe_index, const ImuRawData& imu_raw_data);

 private:
  ImuPreintegration CorrectBias(const ImuPreintegration& imu_preintegration,
                                const Eigen::MatrixXd jacobian,
                                const Eigen::Vector3d delta_accel_bias,
                                const Eigen::Vector3d delta_gyro_bias);
  Eigen::MatrixXd UpdateCovariance(const Eigen::MatrixXd& imu_kinematics,
                                   const Eigen::MatrixXd& noise_kinematics,
                                   const Eigen::MatrixXd& current_covariance,
                                   const double delta_t);
  Eigen::MatrixXd UpdateJacobian(const Eigen::MatrixXd& imu_kinematics,
                                 const Eigen::MatrixXd& current_jacobian,
                                 const double delta_t);

  std::map<int, ImuPreintegration> imu_preintegration_queue_;
  std::map<int, Eigen::MatrixXd> jacobian_queue_;
  std::map<int, Eigen::MatrixXd> covariance_queue_;
};

}  // namespace imu_preintegrator
}  // namespace visual_inertial_slam
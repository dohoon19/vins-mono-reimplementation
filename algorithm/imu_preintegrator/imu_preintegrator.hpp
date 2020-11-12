// image_feature_extractor에서 frame을 받아서 frame_id와 time을 본다.
// frame을 받으면 그전까지 preintegration하던 데이터는 정지한다.
// 해당 frame_id에 preintegration을 넣는다.
// 해당 time부터 preintegration해서 넣는다.
#include <deque>
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
  void SetImuRawData(const ImuRawData& imu_raw_data);
  ImuPreintegration GetImuPreintegration(const int frame_id);
  ImuPreintegration GetImuPreintegrationMeasurementModel(
      const Eigen::Vector3d& current_position,
      const Eigen::Vector3d& current_rotation,
      const Eigen::Vector3d& current_velocity,
      const Eigen::Vector3d& current_accel_bias,
      const Eigen::Vector3d& current_gyro_bias,
      const Eigen::Vector3d& next_position,
      const Eigen::Vector3d& next_accel_bias,
      const Eigen::Vector3d& next_gyro_bias, const double delta_t);
  void Preintegrate(const int frame_id, const ImuRawData& imu_raw_data);

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

  std::deque<ImuRawData> imu_raw_data_buffer_;
  std::map<int, ImuPreintegration> imu_preintegration_queue_;
  std::map<int, Eigen::MatrixXd> jacobian_queue_;
  std::map<int, Eigen::MatrixXd> covariance_queue_;
};

}  // namespace imu_preintegrator
}  // namespace visual_inertial_slam

#ifndef __POSE_ESTIMATOR__
#define __POSE_ESTIMATOR__

#include <map>
#include <set>
#include <unordered_map>

#include "Eigen/Dense"
#include "deque"
#include "opencv2/core.hpp"

namespace visual_inertial_slam {
namespace pose_estimator {

struct Parameters {
  int sliding_window_size = 50;
};

struct Feature {
  int id;
  Eigen::Vector3d on_normalized_image;
  Eigen::Vector2d on_image;
};

struct FeatureInformation {
  int id;
  int tracking_count = 0;
  double depth = 5.0;
  std::set<int> belonged_frame_id_list;
  int GetReferenceFrameIdForDepth() { return *belonged_frame_id_list.begin(); }
};

using FeatureInfoList = std::unordered_map<int, FeatureInformation>;

struct State {
  Eigen::Vector3d translation = {0, 0, 0};
  Eigen::Quaterniond rotation = {1, 0, 0, 0};
};

struct ImuPreintegration {
  double time;
  Eigen::Vector3d translation;
  Eigen::Quaterniond rotation;
  Eigen::Vector3d linear_velocity;
};

struct Imu {
  std::vector<Eigen::Vector3d> time_buffer;
  std::vector<Eigen::Vector3d> linear_acceleration_buffer;
  std::vector<Eigen::Quaterniond> angular_acceleration_buffer;
  Eigen::Vector3d accel_bias;
  Eigen::Vector3d gyro_bias;
  ImuPreintegration preintegration;
};

struct Frame {
  double time;
  int id;
  bool is_keyframe;
  State state;
  std::unordered_map<int, Feature> feature_list;
  Imu imu;
};

class PoseEstimator {
 public:
  explicit PoseEstimator(const Parameters& parameters);
  void SetFrame(const Frame& frame);
  void SetRelocalizedPose(const Eigen::Vector3d& pose);
  std::optional<State> GetPose();
  void Update();

 private:
  void SlideWindow(Frame new_frame, std::map<int, Frame>* frame_window,
                   FeatureInfoList* feature_info_list);
  void OptimizeFramesInWindow(std::map<int, Frame>* frame_window,
                              FeatureInfoList* feature_info_list);
  void MarginalizeOldestFrame(const Frame& new_frame,
                              std::map<int, Frame>* frame_window,
                              FeatureInfoList* feature_info_list);
  // void DiscardLatestFrame(std::deque<Frame>* frame_window,
  // FeatureInfoList* feature_info_list);
  void DiscardOldestFrame(std::map<int, Frame>* frame_window,
                          FeatureInfoList* feature_info_list);
  void Triangulate(std::map<int, Frame>* frame_window,
                   FeatureInformation* feature_info);

  Parameters parameters_;
  std::deque<Frame> frame_buffer_;
  std::map<int, Frame> frame_window_;
  std::deque<Frame> frame_db_;
  FeatureInfoList feature_info_list_;
};

}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#endif  // __POSE_ESTIMATOR__

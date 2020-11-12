#include <chrono>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "image_feature_extractor_ros2_msgs/msg/feature.hpp"
#include "image_feature_extractor_ros2_msgs/msg/frame.hpp"
#include "nav_msgs/msg/path.hpp"
#include "pose_estimator/pose_estimator.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class PoseEstimatorNode : public rclcpp::Node {
 public:
  PoseEstimatorNode() : Node("pose_estimator"), pose_estimator_({}) {
    pose_estimator_ =
        visual_inertial_slam::pose_estimator::PoseEstimator(GetParameters());
    trajectory_.header.frame_id = "map";

    // publisher
    pose_publisher_ =
        create_publisher<geometry_msgs::msg::PoseStamped>("~/pose", 10);
    path_publisher_ = create_publisher<nav_msgs::msg::Path>("~/path", 10);

    // subscriber
    auto frame_callback =
        [this](image_feature_extractor_ros2_msgs::msg::Frame::UniquePtr
                   frame_msg) {
          visual_inertial_slam::pose_estimator::Frame frame;
          visual_inertial_slam::pose_estimator::Feature feature;
          frame.time = frame_msg->time;
          frame.id = frame_msg->id;
          frame.is_keyframe = frame_msg->is_key_frame;
          for (const auto& feature_msg : frame_msg->feature_list) {
            feature.id = feature_msg.id;
            feature.on_image.x() = feature_msg.x_on_image_plane;
            feature.on_image.y() = feature_msg.y_on_image_plane;
            feature.on_normalized_image.x() = feature_msg.x_on_normalized_plane;
            feature.on_normalized_image.y() = feature_msg.y_on_normalized_plane;
            feature.on_normalized_image.z() = 1;
            frame.feature_list.emplace(feature.id, feature);
          }
          pose_estimator_.SetFrame(frame);
          const auto state = pose_estimator_.GetPose();
          std::cerr << "state.has_value() " << state.has_value() << std::endl;
          if (!state.has_value()) return;
          geometry_msgs::msg::PoseStamped pose_stamp_msg;
          pose_stamp_msg.header.frame_id = "map";
          pose_stamp_msg.header.stamp = now();
          pose_stamp_msg.pose.position.x = state.value().translation.x();
          pose_stamp_msg.pose.position.y = state.value().translation.y();
          pose_stamp_msg.pose.position.z = state.value().translation.z();
          pose_stamp_msg.pose.orientation.w = state.value().rotation.w();
          pose_stamp_msg.pose.orientation.x = state.value().rotation.x();
          pose_stamp_msg.pose.orientation.y = state.value().rotation.y();
          pose_stamp_msg.pose.orientation.z = state.value().rotation.z();
          pose_publisher_->publish(pose_stamp_msg);
          trajectory_.poses.emplace_back(pose_stamp_msg);
          path_publisher_->publish(trajectory_);
        };

    frame_subscriber_ =
        create_subscription<image_feature_extractor_ros2_msgs::msg::Frame>(
            "image_feature_extractor/frame", rclcpp::SensorDataQoS(),
            frame_callback);

    auto timer_callback = [this]() { pose_estimator_.Update(); };
    timer_ = create_wall_timer(500ms, timer_callback);
  }

 private:
  visual_inertial_slam::pose_estimator::Parameters GetParameters();
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
  rclcpp::Subscription<image_feature_extractor_ros2_msgs::msg::Frame>::SharedPtr
      frame_subscriber_;
  visual_inertial_slam::pose_estimator::PoseEstimator pose_estimator_;

  std::vector<cv::String> image_path_list_;
  int image_index_ = 0;
  nav_msgs::msg::Path trajectory_;
};

visual_inertial_slam::pose_estimator::Parameters
PoseEstimatorNode::GetParameters() {
  visual_inertial_slam::pose_estimator::Parameters parameters;

  get_parameter_or("sliding_window_size", parameters.sliding_window_size,
                   parameters.sliding_window_size);

  return parameters;
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseEstimatorNode>());
  rclcpp::shutdown();
  return 0;
}

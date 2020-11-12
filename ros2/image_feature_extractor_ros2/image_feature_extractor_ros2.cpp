#include <chrono>
#include <sensor_msgs/msg/image.hpp>

#include "image_feature_extractor/image_feature_extractor.hpp"
#include "image_feature_extractor_ros2_msgs/msg/feature.hpp"
#include "image_feature_extractor_ros2_msgs/msg/frame.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class ImageFeatureExtractorNode : public rclcpp::Node {
 public:
  ImageFeatureExtractorNode()
      : Node("image_feature_extractor"), image_feature_extractor_({}) {
    image_feature_extractor_ =
        visual_inertial_slam::image_feature_extractor::ImageFeatureExtractor(
            GetParameters());

    const std::string image_folder_path =
        "/home/dohoonh/SSLAM/dataset/MH_01_easy/mav0/cam0/data/*.png";
    std::cout << "image_folder_path: " << image_folder_path << std::endl;
    cv::glob(image_folder_path, image_path_list_, false);
    std::cout << "the number of images : " << image_path_list_.size()
              << std::endl;
    if (image_path_list_.size() == 0)
      std::cout << "failed to load images" << std::endl;

    frame_publisher_ =
        create_publisher<image_feature_extractor_ros2_msgs::msg::Frame>(
            "~/frame", rclcpp::SensorDataQoS());

    auto timer_callback = [this]() {
      const auto& image_path = image_path_list_[image_index_++];
      cv::Mat image = imread(image_path, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
        std::cout << "End of Sequence" << std::endl;
        rclcpp::shutdown();
      }
      image_feature_extractor_.SetImage(image);
      image_feature_extractor_.Update();

      const auto result = image_feature_extractor_.GetFrame();
      if (!result.has_value()) return;
      const auto& frame = result.value();
      image_feature_extractor_ros2_msgs::msg::Frame frame_msg;
      image_feature_extractor_ros2_msgs::msg::Feature feature_msg;
      for (size_t i = 0; i < frame.feature_set.Size(); ++i) {
        feature_msg.id = frame.feature_set.GetId(i);
        feature_msg.tracking_count = frame.feature_set.GetTrackingCount(i);
        feature_msg.x_on_image_plane = frame.feature_set.GetFeature(i).x;
        feature_msg.y_on_image_plane = frame.feature_set.GetFeature(i).y;
        feature_msg.x_on_normalized_plane =
            frame.feature_set.GetFeatureOnNormalizedPlane(i).x;
        feature_msg.y_on_normalized_plane =
            frame.feature_set.GetFeatureOnNormalizedPlane(i).y;
        frame_msg.feature_list.emplace_back(feature_msg);
        frame_msg.id = frame.id;
        frame_msg.is_key_frame = frame.is_keyframe;
        frame_msg.time = frame.time;
      }
      frame_publisher_->publish(frame_msg);
    };
    timer_ = create_wall_timer(500ms, timer_callback);
  }

 private:
  visual_inertial_slam::image_feature_extractor::Parameters GetParameters();
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<image_feature_extractor_ros2_msgs::msg::Frame>::SharedPtr
      frame_publisher_;
  visual_inertial_slam::image_feature_extractor::ImageFeatureExtractor
      image_feature_extractor_;

  std::vector<cv::String> image_path_list_;
  int image_index_ = 0;
};

visual_inertial_slam::image_feature_extractor::Parameters
ImageFeatureExtractorNode::GetParameters() {
  visual_inertial_slam::image_feature_extractor::Parameters parameters;

  get_parameter_or("feature_extraction.max_feature_num",
                   parameters.feature_extraction.max_feature_num,
                   parameters.feature_extraction.max_feature_num);
  get_parameter_or("feature_extraction.quality_level",
                   parameters.feature_extraction.quality_level,
                   parameters.feature_extraction.quality_level);
  get_parameter_or("feature_extraction.min_distance",
                   parameters.feature_extraction.min_distance,
                   parameters.feature_extraction.min_distance);
  get_parameter_or("outlier_rejection.maximum_distance_threshold",
                   parameters.outlier_rejection.maximum_distance_threshold,
                   parameters.outlier_rejection.maximum_distance_threshold);
  get_parameter_or("key_frame_threshold.parallax_sum",
                   parameters.key_frame_threshold.parallax_sum,
                   parameters.key_frame_threshold.parallax_sum);
  get_parameter_or("key_frame_threshold.parallax_count",
                   parameters.key_frame_threshold.parallax_count,
                   parameters.key_frame_threshold.parallax_count);
  get_parameter_or("camera.fx", parameters.camera.fx, parameters.camera.fx);
  get_parameter_or("camera.fy", parameters.camera.fy, parameters.camera.fy);
  get_parameter_or("camera.cx", parameters.camera.cx, parameters.camera.cx);
  get_parameter_or("camera.cy", parameters.camera.cy, parameters.camera.cy);
  get_parameter_or("camera.skew_c", parameters.camera.skew_c,
                   parameters.camera.skew_c);
  get_parameter_or("camera.k1", parameters.camera.k1, parameters.camera.k1);
  get_parameter_or("camera.k2", parameters.camera.k2, parameters.camera.k2);
  get_parameter_or("camera.k3", parameters.camera.k3, parameters.camera.k3);
  get_parameter_or("camera.p1", parameters.camera.p1, parameters.camera.p1);
  get_parameter_or("camera.p2", parameters.camera.p2, parameters.camera.p2);
  get_parameter_or("camera.iteration_threshold",
                   parameters.camera.iteration_threshold,
                   parameters.camera.iteration_threshold);
  get_parameter_or("camera.error_threshold", parameters.camera.error_threshold,
                   parameters.camera.error_threshold);

  return parameters;
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageFeatureExtractorNode>());
  rclcpp::shutdown();
  return 0;
}

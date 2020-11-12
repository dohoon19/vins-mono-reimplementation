#include "image_feature_extractor.hpp"

namespace visual_inertial_slam {
namespace image_feature_extractor {

ImageFeatureExtractor::ImageFeatureExtractor(const Parameters& parameters)
    : parameters_(parameters),
      camera_(parameters.camera_parameters),
      frame_id_(0) {}

void ImageFeatureExtractor::SetImage(const cv::Mat& image) {
  image_queue_.emplace_back(image);
}
std::optional<Frame> ImageFeatureExtractor::GetFrame() {
  if (frame_queue_.size() == 0) return std::nullopt;
  const auto frame = frame_queue_.front();
  frame_queue_.pop_front();
  return frame;
}

void ImageFeatureExtractor::Update() {
  if (image_queue_.size() < 2) return;
  const auto previous_image = image_queue_.front();
  image_queue_.pop_front();
  const auto& current_image = image_queue_.front();

  if (previous_feature_set_.Size() == 0) {
    const auto previous_features =
        ExtractFeatures(previous_image, parameters_.max_feature_num);
    for (const auto& feature : previous_features) {
      previous_feature_set_.Push(feature_id_++, 0, feature);
    }
  }

  auto [tracked_previous_feature_set, tracked_current_feature_set] =
      TrackFeatures(previous_image, current_image, previous_feature_set_);

  tracked_previous_feature_set.features =
      Undistort(tracked_previous_feature_set.features);
  tracked_current_feature_set.features =
      Undistort(tracked_current_feature_set.features);
  const auto [inlier_previous_feature_set, inlier_current_feature_set] =
      RejectOutliers(tracked_previous_feature_set, tracked_current_feature_set);

  const int required_feature_counts =
      parameters_.max_feature_num - inlier_current_feature_set.Size();
  const auto additional_features =
      ExtractFeatures(current_image, required_feature_counts);
  FeatureSet current_feature_set = std::move(inlier_current_feature_set);
  for (const auto& feature : additional_features) {
    current_feature_set.Push(feature_id_++, 0, feature);
  }

  Frame frame;
  frame.id = frame_id_++;
  frame.feature_set = current_feature_set;
  frame.is_keyframe =
      IsKeyframe(inlier_previous_feature_set, current_feature_set);
  frame_queue_.emplace_back(frame);

  previous_feature_set_ = std::move(current_feature_set);
}

std::vector<Feature> ImageFeatureExtractor::Undistort(
    const std::vector<Feature> features) {
  std::vector<Feature> undistorted_features;
  for (const auto& feature : features) {
    undistorted_features.emplace_back(camera_.Undistort(feature));
  }
  return undistorted_features;
}

FeatureSetPair ImageFeatureExtractor::RejectOutliers(
    const FeatureSet& previous_feature_set,
    const FeatureSet& current_feature_set) {
  std::vector<uchar> status;
  const double threshold = 3.0;
  const auto F_matrix = cv::findFundamentalMat(
      previous_feature_set.GetFeatures(), current_feature_set.GetFeatures(),
      cv::FM_RANSAC, threshold, 0.99, status);
  FeatureSet inlier_current_feature_set;
  inlier_current_feature_set.Reserve(current_feature_set.Size());
  FeatureSet inlier_previous_feature_set;
  inlier_previous_feature_set.Reserve(previous_feature_set.Size());
  if (status.size() == current_feature_set.Size()) {
    for (size_t i = 0; i < current_feature_set.Size(); ++i) {
      if (!status[i]) continue;
      inlier_current_feature_set.Push(current_feature_set.GetId(i),
                                      current_feature_set.GetTrackingCount(i),
                                      current_feature_set.GetFeature(i));
      inlier_previous_feature_set.Push(previous_feature_set.GetId(i),
                                       previous_feature_set.GetTrackingCount(i),
                                       previous_feature_set.GetFeature(i));
    }
  }

  cv::Mat img;
  image_queue_.front().copyTo(img);
  for (size_t i = 0; i < inlier_current_feature_set.Size(); i++) {
    cv::circle(img, inlier_current_feature_set.GetFeature(i), 8,
               cv::Scalar(255.), 1);
  }
  const std::string window_name = "RejectOutliers";
  cv::namedWindow(window_name, CV_WINDOW_NORMAL);
  cv::imshow(window_name, img);
  cv::waitKey(0);

  return {inlier_previous_feature_set, inlier_current_feature_set};
}

FeatureSetPair ImageFeatureExtractor::TrackFeatures(
    const cv::Mat& previous_image, const cv::Mat& current_image,
    const FeatureSet& previous_feature_set) {
  std::vector<Feature> current_features;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(previous_image, current_image,
                           previous_feature_set.GetFeatures(), current_features,
                           status, err, cv::Size(21, 21), 3);

  FeatureSet tracked_current_feature_set;
  tracked_current_feature_set.Reserve(previous_feature_set.Size());
  FeatureSet tracked_previous_feature_set;
  tracked_previous_feature_set.Reserve(previous_feature_set.Size());

  for (size_t i = 0; i < previous_feature_set.Size(); ++i) {
    if (!status[i]) continue;
    int img_x = cvRound(current_features[i].x);
    int img_y = cvRound(current_features[i].y);
    if (!(0 <= img_x && img_x < previous_image.cols && 0 <= img_y &&
          img_y < previous_image.rows))
      continue;
    tracked_current_feature_set.Push(
        previous_feature_set.GetId(i),
        previous_feature_set.GetTrackingCount(i) + 1, current_features[i]);
    tracked_previous_feature_set.Push(
        previous_feature_set.GetId(i),
        previous_feature_set.GetTrackingCount(i) + 1,
        previous_feature_set.GetFeature(i));
  }

  cv::Mat img;
  previous_image.copyTo(img);
  for (size_t i = 0; i < tracked_current_feature_set.Size(); i++) {
    cv::circle(img, tracked_current_feature_set.GetFeature(i), 8,
               cv::Scalar(255.), 1);
    cv::line(img, tracked_previous_feature_set.GetFeature(i),
             tracked_current_feature_set.GetFeature(i), cv::Scalar(255.), 2);
  }
  const std::string window_name = "TrackFeatures";
  cv::namedWindow(window_name, CV_WINDOW_NORMAL);
  cv::imshow(window_name, img);
  cv::waitKey(0);

  return {tracked_previous_feature_set, tracked_current_feature_set};
}

std::vector<Feature> ImageFeatureExtractor::ExtractFeatures(
    const cv::Mat& image, const int number_of_features) {
  std::vector<Feature> features;
  double quality_level = 0.01;
  double min_distance = 1.0;
  cv::goodFeaturesToTrack(image, features, number_of_features, quality_level,
                          min_distance);

  cv::Mat img;
  image.copyTo(img);
  for (size_t i = 0; i < features.size(); i++) {
    cv::circle(img, features[i], 8, cv::Scalar(255., 0., 0.), 1);
  }
  const std::string window_name = "ExtractFeatures";
  cv::namedWindow(window_name, CV_WINDOW_NORMAL);
  cv::imshow(window_name, img);
  cv::waitKey(0);

  return features;
}

bool ImageFeatureExtractor::IsKeyframe(const FeatureSet& previous_feature_set,
                                       const FeatureSet& current_feature_set) {
  const auto parallax_sum =
      CalculateParallaxSumOfFeatures(previous_feature_set, current_feature_set);
  if (parallax_sum < 0.1) return false;

  auto tracking_count_sum = 0;
  for (size_t i = 0; i < current_feature_set.Size(); ++i) {
    tracking_count_sum += current_feature_set.GetTrackingCount(i);
  }
  if (tracking_count_sum < 100) return false;
  return true;
}

double ImageFeatureExtractor::CalculateParallaxSumOfFeatures(
    const FeatureSet& feature_set_1, const FeatureSet& feature_set_2) {
  double parallax_sum = 0;
  for (size_t i = 0; i < feature_set_1.Size(); ++i) {
    const auto position_difference =
        feature_set_1.GetFeature(i) - feature_set_2.GetFeature(i);
    parallax_sum += std::sqrt(std::pow(2, position_difference.x) +
                              std::pow(2, position_difference.y));
  }
  return parallax_sum;
}

}  // namespace image_feature_extractor
}  // namespace visual_inertial_slam

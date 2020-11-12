#ifndef __IMAGE_FEATURE_EXTRACTOR__
#define __IMAGE_FEATURE_EXTRACTOR__

#include <deque>
#include <iostream>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "camera.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

namespace visual_inertial_slam {
namespace image_feature_extractor {

struct Parameters {
  struct {
    int max_feature_num = 500;
    double quality_level = 0.01;
    double min_distance = 1.0;
  } feature_extraction;
  struct {
    double maximum_distance_threshold = 3.0;
  } outlier_rejection;
  struct {
    double parallax_sum = 0.1;
    int parallax_count = 100;
  } key_frame_threshold;
  camera::Parameters camera;
};

struct FeatureSet {
  void Push(const int id, const int tracking_count, const cv::Point2f& feature,
            const cv::Point2f& feature_on_normalized_plane) {
    ids.emplace_back(id);
    tracking_counts.emplace_back(tracking_count);
    features.emplace_back(feature);
    features_on_normalized_plane.emplace_back(feature_on_normalized_plane);
  }

  void Reserve(const int n) {
    ids.reserve(n);
    tracking_counts.reserve(n);
    features.reserve(n);
  }

  size_t Size() const { return ids.size(); }

  int GetId(const size_t index) const {
    if (index >= Size()) return -1;
    return ids[index];
  }

  int GetTrackingCount(const size_t index) const {
    if (index >= Size()) return -1;
    return tracking_counts[index];
  }

  cv::Point2f GetFeature(const size_t index) const {
    if (index >= Size()) return {-1, -1};
    return features[index];
  }

  cv::Point2f GetFeatureOnNormalizedPlane(const size_t index) const {
    if (index >= Size()) return {-1, -1};
    return features_on_normalized_plane[index];
  }

  std::vector<cv::Point2f> GetFeatures() const { return features; }

  std::vector<cv::Point2f> GetFeaturesOnNormalizedPlane() const {
    return features_on_normalized_plane;
  }

  std::vector<int> ids;
  std::vector<int> tracking_counts;
  std::vector<cv::Point2f> features;
  std::vector<cv::Point2f> features_on_normalized_plane;
};

struct FeatureSetPair {
  FeatureSet previous_features;
  FeatureSet current_features;
};

struct Frame {
  double time = 0;
  int id = 0;
  bool is_keyframe = false;
  FeatureSet feature_set;
};

class ImageFeatureExtractor {
 public:
  explicit ImageFeatureExtractor(const Parameters& parameters);
  void SetImage(const cv::Mat& image);
  std::optional<Frame> GetFrame();
  void Update();

 private:
  FeatureSetPair RejectOutliers(const FeatureSet& previous_features,
                                const FeatureSet& current_features);
  FeatureSetPair TrackFeatures(const cv::Mat& previous_image,
                               const cv::Mat& current_image,
                               const FeatureSet& previous_features);
  std::vector<cv::Point2f> ExtractFeatures(const cv::Mat& current_image,
                                           const int number_of_features);
  bool IsKeyframe(const FeatureSet& previous_feature_set,
                  const FeatureSet& current_feature_set);
  double CalculateParallaxSumOfFeatures(const FeatureSet& frame_1,
                                        const FeatureSet& frame_2);

  Parameters parameters_;
  camera::Camera camera_;
  std::deque<cv::Mat> image_queue_;
  std::deque<Frame> frame_queue_;

  FeatureSet previous_feature_set_;

  int frame_id_;
  int feature_id_;
};

}  // namespace image_feature_extractor
}  // namespace visual_inertial_slam

#endif  // __IMAGE_FEATURE_EXTRACTOR__
#ifndef __POSE_ESTIMATOR__
#define __POSE_ESTIMATOR__

#include <set>
#include <unordered_map>

#include "Eigen/Dense"
#include "deque"
#include "opencv2/core.hpp"

namespace visual_inertial_slam {
namespace pose_estimator {

using Feature = Eigen::Vector2d;

struct FeatureInformation {
  int id;
  int tracked_count;
  std::vector<int> belonged_frame_id;
};

struct Frame {
  double time;
  int id;
  std::unordered_map<int, Feature> features;
};

class PoseEstimator {
 public:
  void SetFrame(const Frame& frame);
  Eigen::Vector3d GetPose();

 private:
  bool IsKeyframe(const Frame& frame);
  double CalculateParallaxOfFeatures(const Frame& frame_1,
                                     const Frame& frame_2);
  std::deque<Frame> frame_buffer_;
  std::deque<Frame> frame_sliding_window_;
};

}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#endif  // __POSE_ESTIMATOR__

#include "pose_estimator.hpp"
namespace visual_inertial_slam {
namespace pose_estimator {
void PoseEstimator::SetFrame(const Frame& frame) {
  frame_buffer_.emplace_back(frame);
}

}  // namespace pose_estimator
}  // namespace visual_inertial_slam

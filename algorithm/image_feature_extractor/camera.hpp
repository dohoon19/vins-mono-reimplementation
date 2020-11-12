#ifndef __CAMERA__
#define __CAMERA__

#include "opencv2/core.hpp"
namespace visual_inertial_slam {
namespace camera {

struct Parameters {
  float fx = 458.654;
  float fy = 457.296;
  float cx = 367.215;
  float cy = 248.375;
  float skew_c = 0;
  float k1 = -0.28340811;
  float k2 = 0.07395907;
  float k3 = 0;
  float p1 = 0.00019359;
  float p2 = 1.76187114e-5;
  int iteration_threshold = 1000;
  double error_threshold = 0.00001;
};

class Camera {
 public:
  explicit Camera(Parameters parameters) : parameters_(parameters) {}
  cv::Point2f Undistort(const cv::Point2f& image_point);
  cv::Point2f Normalize(const cv::Point2f& image_point);

 private:
  Parameters parameters_;
  cv::Point2f Distort(const cv::Point2f& normalized_point);
  cv::Point2f Denormalize(const cv::Point2f& normalized_point);
};

}  // namespace camera
}  // namespace visual_inertial_slam

#endif  // __CAMERA__
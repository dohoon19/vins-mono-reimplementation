#include "pose_estimator.hpp"

#include <iostream>

#include "ceres/ceres.h"
#include "ceres/local_parameterization.h"
#include "ceres/reprojection_error_creator.hpp"

namespace visual_inertial_slam {
namespace pose_estimator {
PoseEstimator::PoseEstimator(const Parameters& parameters)
    : parameters_(parameters) {}
void PoseEstimator::SetFrame(const Frame& frame) {
  std::cerr << "set frame" << std::endl;
  frame_buffer_.emplace_back(frame);
}

void PoseEstimator::Update() {
  if (frame_buffer_.size() == 0) return;
  const auto new_frame = frame_buffer_.front();
  frame_buffer_.pop_front();
  SlideWindow(new_frame, &frame_window_, &feature_info_list_);
  OptimizeFramesInWindow(&frame_window_, &feature_info_list_);
}

void PoseEstimator::OptimizeFramesInWindow(std::map<int, Frame>* frame_window,
                                           FeatureInfoList* feature_info_list) {
  if (frame_window->size() < 3) return;
  // TODO(): create prior error
  ceres::Problem problem;
  auto* huber_function = new ceres::HuberLoss(1.0);

  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  for (auto& [feature_id, feature_information] : *feature_info_list) {
    auto i_frame = feature_information.belonged_frame_id_list.begin();
    if (frame_window->count(*i_frame) == 0) continue;
    const auto& normalized_feature_at_i = frame_window->at(*i_frame)
                                              .feature_list.at(feature_id)
                                              .on_normalized_image;
    auto& i_frame_estimated_feature_depth = feature_information.depth;
    auto& i_frame_translation = frame_window->at(*i_frame).state.translation;
    auto& i_frame_rotation = frame_window->at(*i_frame).state.rotation;
    for (auto j_frame = std::next(i_frame);
         j_frame != feature_information.belonged_frame_id_list.end();
         ++j_frame) {
      if (frame_window->count(*j_frame) == 0) continue;
      const auto& normalized_feature_at_j = frame_window->at(*j_frame)
                                                .feature_list.at(feature_id)
                                                .on_normalized_image;
      auto& j_frame_translation = frame_window->at(*j_frame).state.translation;
      auto& j_frame_rotation = frame_window->at(*j_frame).state.rotation;

      auto* reprojection_error =
          reprojection_error::ReprojectionErrorCreator::Create(
              normalized_feature_at_i, normalized_feature_at_j);

      problem.AddResidualBlock(
          reprojection_error, huber_function, i_frame_translation.data(),
          i_frame_rotation.coeffs().data(), j_frame_translation.data(),
          j_frame_rotation.coeffs().data(), &i_frame_estimated_feature_depth);
      problem.SetParameterization(i_frame_rotation.coeffs().data(),
                                  quaternion_local_parameterization);
      problem.SetParameterization(j_frame_rotation.coeffs().data(),
                                  quaternion_local_parameterization);
    }
  }
  auto& oldest_frame = frame_window->begin()->second;
  problem.SetParameterBlockConstant(oldest_frame.state.translation.data());
  problem.SetParameterBlockConstant(
      oldest_frame.state.rotation.coeffs().data());

  ceres::Solver::Options options;

  int NUM_ITERATIONS = 10000;
  double SOLVER_TIME = 20;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  options.max_solver_time_in_seconds = SOLVER_TIME;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.FullReport() << std::endl;
  std::cerr << "position "
            << frame_window->rbegin()->second.state.translation.x() << ", "
            << frame_window->rbegin()->second.state.translation.y() << ", "
            << frame_window->rbegin()->second.state.translation.z()
            << std::endl;
  const auto euler = frame_window->rbegin()
                         ->second.state.rotation.toRotationMatrix()
                         .eulerAngles(0, 1, 2);
  std::cerr << "rotation " << euler.x() * 180 / M_PI << ", "
            << euler.y() * 180 / M_PI << ", " << euler.z() * 180 / M_PI
            << std::endl;
  // TODO(): marginalize
}

void PoseEstimator::SlideWindow(Frame new_frame,
                                std::map<int, Frame>* frame_window,
                                FeatureInfoList* feature_info_list) {
  if (!frame_window->empty()) {
    new_frame.state.translation =
        frame_window->rbegin()->second.state.translation;
    new_frame.state.rotation = frame_window->rbegin()->second.state.rotation;
  }
  auto result = frame_window->insert({new_frame.id, new_frame});
  assert(result.second);
  for (const auto& pair : new_frame.feature_list) {
    const auto& feature_id = pair.first;
    (*feature_info_list)[feature_id].id = feature_id;
    (*feature_info_list)[feature_id].belonged_frame_id_list.emplace(
        new_frame.id);
  }
  std::cerr << "frame_window->size " << frame_window->size() << std::endl;
  if (static_cast<int>(frame_window->size()) >
      parameters_.sliding_window_size) {
    // const auto latest_frame_in_window = frame_window->back();
    // if (latest_frame_in_window.is_keyframe) {
    //   // MarginalizeOldestFrame(new_frame, frame_window);
    // } else {
    //   DiscardLatestFrame(frame_window, feature_info_list);
    // }
    DiscardOldestFrame(frame_window, feature_info_list);
  }
}

void MarginalizeOldestFrame(const Frame& /*new_frame*/,
                            std::map<int, Frame>* /*frame_window*/,
                            FeatureInfoList* /*feature_info_list*/) {
  // frame_window->pop_front();
}
// void PoseEstimator::DiscardLatestFrame(std::deque<Frame>* frame_window,
//                                        FeatureInfoList* feature_info_list) {
//   auto discarded_frame = frame_window->back();
//   // TODO(): add IMU integration

//   for (const auto& [feature_id, feature_position] :
//        discarded_frame.feature_list) {
//     (void)feature_position;
//     --feature_info_list->at(feature_id).tracking_count;
//     feature_info_list->at(feature_id)
//         .belonged_frame_id_list.erase(discarded_frame.id);
//     if (feature_info_list->at(feature_id).tracking_count < 0)
//       feature_info_list->erase(feature_id);
//   }
//   frame_window_.pop_back();
// }

void PoseEstimator::DiscardOldestFrame(std::map<int, Frame>* frame_window,
                                       FeatureInfoList* feature_info_list) {
  auto discarded_frame = frame_window->begin()->second;
  // TODO(): add IMU integration
  std::cerr << "dbg0" << std::endl;
  for (const auto& [feature_id, feature] : discarded_frame.feature_list) {
    if (feature_info_list->count(feature_id) == 0) continue;
    auto& feature_info = feature_info_list->at(feature_id);
    auto& belonged_frame_id_list = feature_info.belonged_frame_id_list;
    belonged_frame_id_list.erase(discarded_frame.id);
    if (belonged_frame_id_list.size() > 4) {
      auto& next_frame = std::next(frame_window->begin())->second;
      const Eigen::Vector3d feature_at_discarded_frame =
          feature.on_normalized_image * feature_info.depth;
      const Eigen::Vector3d feature_at_world_frame =
          discarded_frame.state.rotation * feature_at_discarded_frame +
          discarded_frame.state.translation;
      const Eigen::Vector3d feature_at_next_frame =
          next_frame.state.rotation.inverse() *
          (feature_at_world_frame - next_frame.state.translation);
      feature_info.depth = feature_at_next_frame.z();
    } else {
      feature_info_list->erase(feature_id);
    }
  }
  std::cerr << "dbg1" << std::endl;
  frame_window->erase(frame_window->begin());
}

std::optional<State> PoseEstimator::GetPose() {
  if (frame_window_.size() == 0) return {};
  return frame_window_.rbegin()->second.state;
}

void PoseEstimator::Triangulate(std::map<int, Frame>* frame_window,
                                FeatureInformation* feature_info) {
  const auto& belonged_frame_id_list = feature_info->belonged_frame_id_list;
  if (belonged_frame_id_list.size() < 3) return;
  auto ref_frame_id = belonged_frame_id_list.begin();
  const auto& ref_frame = frame_window->at(*ref_frame_id);

  Eigen::MatrixXd svd_A(2 * belonged_frame_id_list.size(), 4);

  Eigen::Matrix<double, 3, 4> SE3_ref;
  Eigen::Vector3d ref_translation = ref_frame.state.translation;
  Eigen::Matrix3d ref_rotation = ref_frame.state.rotation.toRotationMatrix();
  SE3_ref.leftCols<3>() = Eigen::Matrix3d::Identity();
  SE3_ref.rightCols<1>() = Eigen::Vector3d::Zero();

  int svd_idx = 0;
  for (auto j_frame_id = ref_frame_id;
       j_frame_id != belonged_frame_id_list.end(); ++j_frame_id) {
    if (frame_window->count(*j_frame_id) == 0) continue;
    const auto& j_frame = frame_window->at(*j_frame_id);
    const auto& j_translation = j_frame.state.translation;
    const auto j_rotation = j_frame.state.rotation.toRotationMatrix();

    Eigen::Vector3d translation =
        ref_rotation.transpose() * (j_translation - ref_translation);
    Eigen::Matrix3d rotation = ref_rotation.transpose() * j_rotation;
    Eigen::Matrix<double, 3, 4> SE3;
    SE3.leftCols<3>() = rotation.transpose();
    SE3.rightCols<1>() = -rotation.transpose() * translation;
    Eigen::Vector3d feature_position =
        j_frame.feature_list.at(feature_info->id).on_normalized_image;
    svd_A.row(svd_idx++) =
        feature_position[0] * SE3.row(2) - feature_position[2] * SE3.row(0);
    svd_A.row(svd_idx++) =
        feature_position[1] * SE3.row(2) - feature_position[2] * SE3.row(1);
  }
  assert(svd_idx == svd_A.rows());
  Eigen::Vector4d svd_V =
      Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
          .matrixV()
          .rightCols<1>();
  double depth = svd_V[2] / svd_V[3];
  feature_info->depth = depth;

  feature_info->depth = depth < 0.1 ? 5.0 : depth;
}

}  // namespace pose_estimator
}  // namespace visual_inertial_slam

#include <iostream>

#include "image_feature_extractor.hpp"

using namespace visual_inertial_slam::image_feature_extractor;
int main(int argc, char** argv) {
  const Parameters parameters = {};
  ImageFeatureExtractor image_feature_extractor(parameters);
  const std::string default_image_folder_path =
      "/home/dohoonh/SSLAM/dataset/MH_01_easy/mav0/cam0/data/"
      "*.png";
  const std::string image_folder_path =
      (argc == 2) ? argv[1] : default_image_folder_path;
  std::cout << "image_folder_path: " << image_folder_path << std::endl;
  std::vector<cv::String> image_path_list;
  cv::glob(image_folder_path, image_path_list, false);
  std::cout << "the number of images : " << image_path_list.size() << std::endl;
  if (image_path_list.size() == 0)
    std::cout << "failed to load images" << std::endl;

  for (const auto& image_path : image_path_list) {
    cv::Mat image = imread(image_path, cv::IMREAD_GRAYSCALE);
    std::cerr << "image_path: " << image_path << std::endl;
    if (image.empty()) {
      std::cout << "End of Sequence" << std::endl;
      break;
    }
    image_feature_extractor.SetImage(image);
    image_feature_extractor.Update();
  }

  return 0;
}

#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include <vector>
#include <aligner.hpp>

#include <opencv2/opencv.hpp>

#define EC_MC_Y 48.0
#define EC_Y 40.0
#define IMG_SIZE 128

cv::Mat align_eyes(const cv::Mat &, cv::Point &, cv::Point &, cv::Point &);

cv::Mat resize(const cv::Mat &, cv::Point &, cv::Point &);

cv::Mat crop(const cv::Mat &, cv::Point &, cv::Point &);

cv::Mat normalize(const cv::Mat &, const std::vector<cv::Point> &);

cv::Mat normalize_depth(Aligner *aligner);

#endif

#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <caffe/caffe.hpp>
#include <igl/sort.h>
#include <igl/find.h>

void convertToMatrix(caffe::Blob<float> * prob, caffe::Blob<float> * conv, Eigen::MatrixXd & map, std::vector<Eigen::MatrixXd> & reg);

void convertToVector(caffe::Blob<float> * prob, std::vector<double> & score);

void filter(Eigen::MatrixXd & total_boxes, Eigen::VectorXi & pass_t, Eigen::MatrixXd & score);

void filter(Eigen::MatrixXd & total_boxes, std::vector<int> & pass_t, std::vector<double> & score);

void getMV(caffe::Blob<float> * conv, Eigen::MatrixXd & mv, std::vector<int> & pass_t);

Eigen::VectorXi _find(Eigen::MatrixXd A, Eigen::MatrixXd B);

Eigen::VectorXi _find(Eigen::MatrixXd A, double b);

void _find(std::vector<double> & A, double b, std::vector<int> & C);

void _fix(Eigen::MatrixXd & M);

Eigen::MatrixXd subOneRow(Eigen::MatrixXd M, int index);

Eigen::MatrixXd subOneRowRerange(Eigen::MatrixXd & M, std::vector<int> & I);

void npwhere_vec(std::vector<int> & index, const std::vector<double> & value, const double threshold);

void _select(Eigen::MatrixXd &src, Eigen::MatrixXd &dst, const std::vector<int> & pick);

void bbreg(Eigen::MatrixXd & boundingbox, Eigen::MatrixXd & reg);

void pad(Eigen::MatrixXd & boundingbox, double w, double h, Eigen::MatrixXd & result);

void rerec(Eigen::MatrixXd & boundingbox);

void nms(Eigen::MatrixXd & boundingbox, float threshold, const std::string & type, std::vector<int> & pick);

//void _stage3(cv::Mat & img_mat, std::shared_ptr<caffe::Net<float>> ONet, std::vector<float> & threshold, Eigen::MatrixXd & total_boxes);

#endif

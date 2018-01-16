#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>

#include <Eigen/Dense>

#include <igl/sort.h>
#include <igl/find.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <caffe/caffe.hpp>

#include <camera.hpp>

#define MTCNN_MODEL_PATH "reqs/model"
#define DEPTH_CTHRESHOLD 1200
#define STEP 6

class Detector {
public:
    // results
    std::vector<cv::Rect> detections;
    std::vector<std::vector<cv::Point>> landmarks;

	// mtcnn
	caffe::Net<float> PNet, RNet, ONet;
    const std::vector<float> thresholds;
    const float factor;
    const int minsize;
    const bool fastresize;
    const cv::Point3d face_half_size;
    
    Eigen::MatrixXd stage1(cv::Mat &);
    void prepare_data1(const cv::Mat &);
    void stage2(cv::Mat &, Eigen::MatrixXd &);
    void stage3(cv::Mat &, Eigen::MatrixXd &);
    void prepare_data2(const caffe::Net<float> &, const std::vector<cv::Mat> &);
    void generate_bounding_box(Eigen::MatrixXd &, std::vector<Eigen::MatrixXd> &, double, double, Eigen::MatrixXd &);
    
    // depth detection
    cv::Point3d best_face_coords;
    CvHaarClassifierCascade * face_cascade;
	IplImage * p, * m, * sum, * sqsum, * tiltedsum, * msum, * sumint, * tiltedsumint;
	const int proj_height, proj_width;
	const cv::Point proj_center;
	const int min_x, max_x, min_y, max_y;
	
	void compute_projection(const std::vector<cv::Point3d> &, double m[3][3], const double b = DEPTH_CTHRESHOLD);
	void compute_rotation_matrix(double m[3][3], double i[3][3], const double, const double, const double);
	cv::Point3d depth_landmark(const cv::Point &, const Camera &);
	std::vector<cv::Point3d> grid_sampling(const cv::Rect &, const Camera &, const double thr = DEPTH_CTHRESHOLD, const int jump = STEP);
	bool depth_verification(const std::vector<cv::Point3d> &);


    Detector(const std::string & model_path = MTCNN_MODEL_PATH, const bool fast_resize = false);
    ~Detector();
    Detector(Detector &&) = delete;
    Detector(const Detector &) = delete;
    Detector & operator=(const Detector &) = delete;
    
    void operator()(const cv::Mat &);
    bool is_face(const unsigned int, const Camera &);
    unsigned int num_faces() const;
    const cv::Rect & get_detection(const unsigned int) const;
    const cv::Point3d & get_depth_detection() const;
    const std::vector<cv::Rect> & get_detections() const;
    const std::vector<cv::Point> & get_landmarks(const unsigned int) const;
    const std::vector<std::vector<cv::Point>> & get_all_landmarks() const;
};

#endif

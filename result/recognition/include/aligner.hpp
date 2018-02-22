#ifndef ALIGN_HPP
#define ALIGN_HPP

#include <vector>
#include <limits>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <camera.hpp>

#define ROI_WIDTH 0.8
#define ROI_HEIGHT 0.6
#define MODEL_WIDTH 48.0
#define MODEL_HEIGHT_1 56.0
#define MODEL_HEIGHT_2 16.0
#define MODEL_RESOLUTION 1.0

#define MAX_ICP_ITERATIONS 400
#define OUTLIER_SQUARED_THRESHOLD 225.0
#define OUTLIER_THRESHOLD 15.0

#define DEPTH_THRESHOLD 2000

#define AVERAGE_MODEL_PATH "misc/average.dat"

class Aligner {
private:
    std::vector<cv::Point3d> ninja_mask_cloud;
    cv::Vec6d transf;
    cv::Mat avg;
    int row, col;
    const cv::Point proj_origin, final_proj_origin;
    const int width, height;
    
    void load_avg_face_model();
    void transpose_matrix(double [3][3], double [3][3]) const;
    void sqrt_3x3_matrix(double [3][3], double [3][3]) const;
    void invert_3x3_matrix(double [3][3], double [3][3]) const;
    void prep_ninja_mask(const std::vector<cv::Point3d> &, const cv::Point3d &);
    void find_pre_alignment();
    void compute_depth_map(const std::vector<cv::Point3d> &);
    void compute_registered_depth_map(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &);
    void compute_rotation_matrix(double [3][3], double, double, double);
    cv::Point3d transform_point(double [3][3], const cv::Point3d &, const cv::Point3d &);
    void horn(cv::Vec6d &, const std::vector<cv::Point3d> &, const std::vector<cv::Point3d> &);
    void hole_filling(cv::Mat &, cv::Mat &);
    bool find_alignment(const std::vector<cv::Point3d> &, const cv::Point3d &);
    double iterative_closest_points(const int max_iterations = MAX_ICP_ITERATIONS, const double outl_sqrd_thr = 225.0);
    double project_and_walk(const cv::Point3d &);
    double walk(double, double, double, int, int, int, double);

public:
    Aligner();
    Aligner(const std::vector<cv::Point3d> &, const cv::Point3d &);
    Aligner(const Camera &, const cv::Vec6d &, const std::vector<cv::Point> &);
    ~Aligner() {}
    Aligner(const Aligner &) = delete;
    Aligner(Aligner &&) = delete;
    Aligner operator = (const Aligner &) = delete;
    
    cv::Mat img;
    std::vector<cv::Point2f> landmarks, ir_landmarks;

    void align(const std::vector<cv::Point3d> &, const cv::Point3d &);
    void align_registered(const std::vector<cv::Point3d> &, const std::vector<cv::Point2d> &, const cv::Point3d &);
    void align_registered(const Camera &, const cv::Vec6d &, const std::vector<cv::Point> &);
};

#endif

#ifndef LOAD_REALSENSE_HPP
#define LOAD_REALSENSE_HPP

#include "camera.hpp"
#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <string>
#include <lzf.h>

class FileHandler {
private:
    std::ifstream file;
    uint64_t compressed_size, uncompressed_size;
    unsigned long long int size;
    std::vector<uint8_t> compressed, uncompressed;
    struct timeval current_time;
    bool is_compressed;
    size_t elem_size;
    cv::Mat &img;

public:
    FileHandler(const std::string &, cv::Mat &);
    ~FileHandler() {}
    FileHandler(const FileHandler &) = delete;
    FileHandler(FileHandler &&) = delete;
    FileHandler & operator=(const FileHandler &) = delete;

    int width, height, type;
    bool nextFrame();
    void open(const std::string &);
};

class Load : public Camera {
private:
    const float depth_q, color_q;
    rs::intrinsics depth_intrin, color_intrin;
	rs::extrinsics depth_to_color, depth_to_ir;
    float scale;
    FileHandler depth_file, rgb_file, ir_file;
    std::vector<double> i_to_y, j_to_x;
    std::vector< std::vector<cv::Point2i> > undistortion_map;

public:
    Load(const std::string &);
    virtual ~Load();
    Load(const Load &) = delete;
    Load(Load &&) = delete;
    Load & operator=(const Load &) = delete;

    void getImages();
    void load(const std::string &);
    cv::Point3d depthToXyz(const cv::Point &) const;
    cv::Point3d depthToXyz(int, int) const;
    cv::Point2d xyzToColor(const cv::Point3d &) const;
    cv::Point2d xyzToDepth(const cv::Point3d &) const;
    void writeParams(const std::string &);
    bool shutdown;

};

#endif /* end of include guard: LOAD_REALSENSE_HPP */

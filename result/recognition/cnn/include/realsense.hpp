#ifndef REALSENSE_HPP
#define REALSENSE_HPP

#include "camera.hpp"
#include <librealsense/rs.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#define RGB_HEIGHT 720
#define RGB_WIDTH 1280
#define D_HEIGHT 480
#define D_WIDTH 640
#define IR_HEIGHT 480
#define IR_WIDTH 640

class RealSense : public Camera {
private:
    rs::device *dev;
    rs::context ctx;
    rs::intrinsics depth_intrin, color_intrin;
    rs::extrinsics depth_to_color, depth_to_ir;
    float scale;
    const int rgb_width, rgb_height;
    void *depth_frame, *color_frame, *infrared_frame;
    
    template<int camera_model>
    void setup();

public:
    RealSense(int camera_idx = 0, int arg_rgb_width = RGB_WIDTH, int arg_rgb_height = RGB_HEIGHT);
    ~RealSense();
    RealSense(RealSense &&) = delete;
    RealSense(const RealSense &) = delete;
    RealSense & operator=(const RealSense &) = delete;

    void getImages();
    void writeParams(const std::string &);
    cv::Point3d depthToXyz(const cv::Point &) const;
    cv::Point3d depthToXyz(int, int) const;
    cv::Point2d xyzToColor(const cv::Point3d &) const;
    cv::Point2d xyzToDepth(const cv::Point3d &) const;

};

#endif

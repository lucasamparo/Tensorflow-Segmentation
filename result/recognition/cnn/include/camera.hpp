#ifndef CAMERA
#define CAMERA

#include <opencv2/opencv.hpp>
#include <string>

class Camera {
public:
    virtual void getImages() = 0;
    virtual cv::Point3d depthToXyz(const cv::Point &) const = 0;
    virtual cv::Point3d depthToXyz(int, int) const = 0;
    virtual cv::Point2d xyzToColor(const cv::Point3d &) const = 0;
    virtual cv::Point2d xyzToDepth(const cv::Point3d &) const = 0;
    virtual void writeParams(const std::string &) = 0;
    cv::Mat depth, color, ir;
};
#endif

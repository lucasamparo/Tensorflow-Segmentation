#ifndef DESCRIPTOR_HPP
#define DESCRIPTOR_HPP

#include <opencv2/opencv.hpp>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

class Descriptor {
private:
    caffe::Net<float> net;
    cv::Mat input, output;

public:
    Descriptor();
    ~Descriptor() = default;
    Descriptor(const Descriptor &) = delete;
    Descriptor(Descriptor &&) = delete;
    Descriptor & operator=(const Descriptor &) = delete;
    
    const cv::Mat & operator()(const cv::Mat &);
    static double compare(const cv::Mat &, const cv::Mat &);
};

#endif

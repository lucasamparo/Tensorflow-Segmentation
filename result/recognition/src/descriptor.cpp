#include <descriptor.hpp>

#define MODEL_PROTO_PATH "reqs/model/LightenedCNN_C_deploy.prototxt"
#define MODEL_BIN_PATH "reqs/model/LightenedCNN_C_deploy.caffemodel"

Descriptor::Descriptor() : net(MODEL_PROTO_PATH, caffe::TEST) {
    net.CopyTrainedLayersFrom(MODEL_BIN_PATH);
    
    caffe::Blob<float> & input_blob = *net.blob_by_name("data");
    caffe::Blob<float> & output_blob = *net.blob_by_name("eltwise_fc1");
    input = cv::Mat(input_blob.height(), input_blob.width(), CV_32F, input_blob.mutable_cpu_data());
    output = cv::Mat(output_blob.shape(1), output_blob.shape(0), CV_32F, output_blob.mutable_cpu_data());
}

double Descriptor::compare(const cv::Mat & a, const cv::Mat & b) {
    return a.dot(b) / cv::norm(a) / cv::norm(b);
}

const cv::Mat & Descriptor::operator()(const cv::Mat & img) {
	img.convertTo(input, CV_32F, 1.0/255.0);
    net.Forward();
    return output;
}

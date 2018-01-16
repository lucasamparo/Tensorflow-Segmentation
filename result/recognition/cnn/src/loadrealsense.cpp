#include "loadrealsense.hpp"
#include "liblzf/lzf_d.c"

FileHandler::FileHandler(const std::string & filename, cv::Mat &img_ref) : img(img_ref) {
    file.open(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Unable to open file " << filename << std::endl;
        throw 1;
    }
    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));
    file.read(reinterpret_cast<char *>(&type), sizeof(int));
    file.read(reinterpret_cast<char *>(&elem_size), sizeof(size_t));
    size = width*height*elem_size;
    uncompressed.resize(size);
    img.create(height, width, type);
}

void FileHandler::open(const std::string & filename) {
    if (file.is_open()) file.close();
    file.open(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Unable to open file " << filename << std::endl;
        throw 1;
    }
    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));
    file.read(reinterpret_cast<char *>(&type), sizeof(int));
    file.read(reinterpret_cast<char *>(&elem_size), sizeof(size_t));
    size = width*height*elem_size;
    uncompressed.resize(size);
    img.create(height, width, type);
}

Load::Load(const std::string &path) : depth_file(path + "depth.log", depth), rgb_file(path + "color.log", color), ir_file(path + "ir.log", ir), depth_q(0.01), color_q(0.002199) {
    std::ifstream params_file(path + "params.log", std::ios::in | std::ios::binary);
    if (!params_file.is_open()) {
        std::cout << "Unable to open file " << path << "params.log" << std::endl;
        throw 1;
    }
    shutdown = false;

    params_file.read(reinterpret_cast<char *>(&depth_to_color), sizeof(depth_to_color));
    params_file.read(reinterpret_cast<char *>(&depth_to_ir), sizeof(depth_to_ir));
    params_file.read(reinterpret_cast<char *>(&depth_intrin), sizeof(depth_intrin));
	params_file.read(reinterpret_cast<char *>(&color_intrin), sizeof(color_intrin));
    params_file.read(reinterpret_cast<char *>(&scale), sizeof(scale));
}

Load::~Load() {}

void Load::load(const std::string &path) {
    depth_file.open(path + "depth.log");
    rgb_file.open(path + "color.log");
    ir_file.open(path + "ir.log");
    shutdown = false;
}

bool FileHandler::nextFrame() {
    file.read(reinterpret_cast<char *>(&current_time), sizeof(struct timeval));
    file.read(reinterpret_cast<char *>(&compressed_size), sizeof(uint64_t));

    if (compressed_size <= 0 || compressed_size > size) {
        file.read(reinterpret_cast<char *>(&uncompressed[0]), size);
    }
    else {
        compressed.resize(compressed_size);
        file.read(reinterpret_cast<char *>(&compressed[0]), compressed_size);
        lzf_decompress(&compressed[0], compressed_size, &uncompressed[0], uncompressed.size());
    }

    img.data = &uncompressed[0];

    if (file)
        return true;

    return false;
}

void Load::getImages() {
    if (!(depth_file.nextFrame() && rgb_file.nextFrame() && ir_file.nextFrame()))
        shutdown = true;
}

cv::Point3d Load::depthToXyz(const cv::Point &coord) const {
    if (coord.y >= 0 && coord.y < depth.rows && coord.x >= 0 && coord.x < depth.cols) {
    	const float depth_in_meters = depth.at<unsigned short>(coord.y, coord.x) * scale;
    	const rs::float2 depth_pixel = {(float)coord.x, (float)coord.y};
    	const rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
    	return cv::Point3d(depth_point.x*1000.0, -depth_point.y*1000.0, -depth_point.z*1000.0);
    }
    return cv::Point3d(0, 0, 0);
}

cv::Point3d Load::depthToXyz(int i, int j) const {
    if (i >= 0 && i < depth.rows && j >= 0 && j < depth.cols) {
        const float depth_in_meters = depth.at<unsigned short>(i, j) * scale;
    	const rs::float2 depth_pixel = {(float)j, (float)i};
    	const rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
        return cv::Point3d(depth_point.x*1000.0, -depth_point.y*1000.0, -depth_point.z*1000.0);
    }
	return cv::Point3d(0, 0, 0);
}

cv::Point2d Load::xyzToColor(const cv::Point3d &point) const {
    const rs::float3 depth_point = {(float)(point.x/1000.0), (float)(-point.y/1000.0), (float)(-point.z/1000.0)};
    const rs::float3 color_point = depth_to_color.transform(depth_point);
    const rs::float2 color_pixel = color_intrin.project(color_point);
    return cv::Point2d(color_pixel.x, color_pixel.y);
}

cv::Point2d Load::xyzToDepth(const cv::Point3d &point) const {
    cv::Point2d p;
    p.x = -(point.x * depth_intrin.fx)/point.z + depth_intrin.ppx;
    p.y = (point.y * depth_intrin.fy)/point.z + depth_intrin.ppy;
    return p;
}

void Load::writeParams(const std::string & s) {
    // empty. exists only to qualify load as a camera
}

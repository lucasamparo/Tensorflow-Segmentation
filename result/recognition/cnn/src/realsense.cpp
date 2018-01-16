#include "realsense.hpp"

template<>
void RealSense::setup<0>() {
    /* set camera parameters */
    dev->set_option((rs::option)RS_OPTION_F200_CONFIDENCE_THRESHOLD, 6); // [0-15] # 0 means HIGH noise and LOW data loss
    dev->set_option((rs::option)RS_OPTION_F200_FILTER_OPTION, 5); // [0-7] # 0 means no filter
    dev->set_option((rs::option)RS_OPTION_F200_MOTION_RANGE, 0); // [0-100] # 100 means LOW fps and LOW data loss
    dev->set_option((rs::option)RS_OPTION_F200_LASER_POWER, 16);
    dev->set_option((rs::option)RS_OPTION_F200_ACCURACY, 3);
}

template<>
void RealSense::setup<1>() {
    /* set camera parameters */
    // dev->set_option((rs::option)0, 0);
    // dev->set_option((rs::option)1, 0);
    // dev->set_option((rs::option)2, 50);
    // dev->set_option((rs::option)4, 64);
    // dev->set_option((rs::option)5, 300);
    // dev->set_option((rs::option)6, 0);
    // dev->set_option((rs::option)7, 64);
    // dev->set_option((rs::option)8, 50);
    // dev->set_option((rs::option)9, 4600);
    // dev->set_option((rs::option)11, 1);
    // dev->set_option((rs::option)12, 16);
    // dev->set_option((rs::option)13, 1);
    // dev->set_option((rs::option)15, 5);
    // dev->set_option((rs::option)16, 3);
    // dev->set_option((rs::option)18, 1);
    // dev->set_option((rs::option)19, 1);
    // dev->set_option((rs::option)20, 180);
    // dev->set_option((rs::option)21, 605);
    // dev->set_option((rs::option)22, 303);
    // dev->set_option((rs::option)23, 2);
    // dev->set_option((rs::option)24, 16);
    // dev->set_option((rs::option)25, -1);
    // dev->set_option((rs::option)26, 1250);
    // dev->set_option((rs::option)27, 650);
    // dev->set_option((rs::option)33, 0);
    // dev->set_option((rs::option)34, 0);
}

RealSense::RealSense(int camera_idx, int arg_rgb_width, int arg_rgb_height) : rgb_width(arg_rgb_width), rgb_height(arg_rgb_height) {
    if (ctx.get_device_count() <= camera_idx) {
        std::cerr << "Error: not enough devices connected" << std::endl;
        throw 1;
    }

    /* open device */
	dev = ctx.get_device(camera_idx);

    depth.create(D_HEIGHT, D_WIDTH, CV_16UC1);
    color.create(rgb_height, rgb_width, CV_8UC3);
    ir.create(IR_HEIGHT, IR_WIDTH, CV_8UC1);

    std::string camera_model(dev->get_name());
    const int fps = 30;
    if (camera_model == "Intel RealSense F200") {
        setup<0>();
    }
    else if (camera_model == "Intel RealSense SR300") {
        setup<1>();
    }

	/* configure streams */
	dev->enable_stream(rs::stream::depth, D_WIDTH, D_HEIGHT, rs::format::z16, fps);
	dev->enable_stream(rs::stream::color, rgb_width, rgb_height, rs::format::bgr8, fps);
	dev->enable_stream(rs::stream::infrared, IR_WIDTH, IR_HEIGHT, rs::format::y8, fps);
    dev->start();

    /* fetch camera parameters */
    depth_intrin = dev->get_stream_intrinsics(rs::stream::depth);
    color_intrin = dev->get_stream_intrinsics(rs::stream::color);
    depth_to_ir = dev->get_extrinsics(rs::stream::depth, rs::stream::infrared);
    depth_to_color = dev->get_extrinsics(rs::stream::depth, rs::stream::color);
    scale = dev->get_depth_scale();
}

RealSense::~RealSense() {
    dev->stop();
}

void RealSense::getImages() {
    dev->wait_for_frames();

    /* access frames using opencv */
    depth_frame = (void *) (dev->get_frame_data(rs::stream::depth));
    color_frame = (void *) (dev->get_frame_data(rs::stream::color));
    infrared_frame = (void *) (dev->get_frame_data(rs::stream::infrared));

    depth = cv::Mat(D_HEIGHT, D_WIDTH, CV_16UC1, (void *) depth_frame);
    color = cv::Mat(rgb_height, rgb_width, CV_8UC3, (void *) color_frame);
    ir = cv::Mat(IR_HEIGHT, IR_WIDTH, CV_8UC1, (void *) infrared_frame);
}

cv::Point3d RealSense::depthToXyz(const cv::Point &coord) const {
    if (coord.y >= 0 && coord.y < depth.rows && coord.x >= 0 && coord.x < depth.cols) {
    	const float depth_in_meters = depth.at<unsigned short>(coord.y, coord.x) * scale;
    	const rs::float2 depth_pixel = {(float)coord.x, (float)coord.y};
    	const rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
    	return cv::Point3d(depth_point.x*1000.0, -depth_point.y*1000.0, -depth_point.z*1000.0);
    }
    return cv::Point3d(0, 0, 0);
}

cv::Point3d RealSense::depthToXyz(int i, int j) const {
    if (i >= 0 && i < depth.rows && j >= 0 && j < depth.cols) {
        const float depth_in_meters = depth.at<unsigned short>(i, j) * scale;
    	const rs::float2 depth_pixel = {(float)j, (float)i};
    	const rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
        return cv::Point3d(depth_point.x*1000.0, -depth_point.y*1000.0, -depth_point.z*1000.0);
    }
	return cv::Point3d(0, 0, 0);
}

cv::Point2d RealSense::xyzToColor(const cv::Point3d &point) const {
    const rs::float3 depth_point = {(float)(point.x/1000.0), (float)(-point.y/1000.0), (float)(-point.z/1000.0)};
    const rs::float3 color_point = depth_to_color.transform(depth_point);
    const rs::float2 color_pixel = color_intrin.project(color_point);
    return cv::Point2d(color_pixel.x, color_pixel.y);
}

cv::Point2d RealSense::xyzToDepth(const cv::Point3d &point) const {
    cv::Point2d p;
    p.x = -(point.x * depth_intrin.fx)/point.z + depth_intrin.ppx;
    p.y = (point.y * depth_intrin.fy)/point.z + depth_intrin.ppy;
    return p;
}

void RealSense::writeParams(const std::string & path) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(&depth_to_color), sizeof(depth_to_color));
    file.write(reinterpret_cast<const char *>(&depth_to_ir), sizeof(depth_to_ir));
    file.write(reinterpret_cast<const char *>(&depth_intrin), sizeof(depth_intrin));
	file.write(reinterpret_cast<const char *>(&color_intrin), sizeof(color_intrin));
    file.write(reinterpret_cast<const char *>(&scale), sizeof(scale));
}

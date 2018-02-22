#include "realsense.hpp"
#include "capture.hpp"
#include <string>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#define POST_FRONTAL_FACE_RECORD_TIME 3.0
#define RECORDING_PATH "recording/"

using namespace std::chrono;
using namespace cv;
using namespace std;

int stopFlag = 0;

void handler( int ) {
    stopFlag = 1;
}

int main(int argc, char const *argv[])
{
    RealSense video;
	video.getImages();
    /* create folder for the current recording and name */
    const std::string current_rec_path = RECORDING_PATH + std::string(argv[1]);
    int status2 = system(("mkdir -p " + current_rec_path).c_str());
    /* save the camera parameters */
	video.writeParams(current_rec_path + "/params.log");
    /* open recording files */
    BufferHandler depth_buf(video.depth, current_rec_path + "/depth.log"), rgb_buf(video.color, current_rec_path + "/color.log"), ir_buf(video.ir, current_rec_path + "/ir.log");
    /* record three seconds while the user leaves */
    signal(SIGINT, &handler);
    std::ofstream file(current_rec_path + "/timestamps.log");
    std::time_t start = std::time(NULL);
    int count = 0;
    float start_record = getTickCount();
    float now_record = getTickCount();
    while (!stopFlag && (now_record - start_record) / getTickFrequency() <= 300) { //300 s = 5 min
        depth_buf.copy(video.depth.data);
        rgb_buf.copy(video.color.data);
        ir_buf.copy(video.ir.data);
        video.getImages();
        milliseconds ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        std::time_t now = std::time(NULL);
        file << std::to_string(ms.count()) << std::endl;
        count++;
        now_record = getTickCount();
        // std::cout << std::to_string(ms.count()) << std::endl;
    }
    std::time_t end = std::time(NULL) - start;
    file << "FPS " << (double)count / end;
    /* finish recording */
    depth_buf.join();
	rgb_buf.join();
	ir_buf.join();
	return 0;
}
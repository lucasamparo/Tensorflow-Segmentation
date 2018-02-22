#include "loadrealsense.hpp"
#include <normalize.hpp>
#include <detector.hpp>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include <aligner.hpp>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) try {
    if (argc <= 1) {
        std::cout << "Insufficient arguments. Usage: pass the recording directory as a parameter" << std::endl;
        return 1;
    }
    std::vector<std::string> args(argv + 1, argv + argc);
    Load video(args[0]);
    Detector face_detector;
    while (!video.shutdown) {
        video.getImages();
        Mat color = video.color;
        face_detector(color);
        if(face_detector.num_faces() == 1) {
            std::vector<cv::Point> color_landmarks = face_detector.get_landmarks(0);
            cvtColor(color, color, CV_BGR2GRAY);
            Mat normalized_face = normalize(color, color_landmarks);
            ifstream file_inf(args[0]+"params.log"); //inicializar com info.log
            cv::Vec6d transf;
            if(file_inf.is_open())
                file_inf.read(reinterpret_cast<char *>(&transf), sizeof(transf));
            else {
                std::cerr << "Erro: nao foi possivel abrir o arquivo " << args[0]+"params.log" << std::endl;
                return 1;
            }
            Aligner aligner(video, transf, color_landmarks);
            Mat depth_norm_face = normalize_depth(&aligner);
            imshow("Normalizada", depth_norm_face);
        }
        // cv::imshow("color", video.color);
        // cv::imshow("ir", video.ir);
        cv::imshow("depth", video.depth);
        cv::waitKey(10);
    }

    return 0;
}
catch (...) {
    return 1;
}

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <detector.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>

using namespace cv;
using namespace cv::face;
using namespace std;

Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    switch(src.channels()) {
	    case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	    case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	    default:
		src.copyTo(dst);
		break;
    }
    return dst;
}

vector<string> loadDataset(string p){
    	DIR *dir = NULL;
	struct dirent *drnt = NULL;
	char * pch;
	vector<string> path;

	dir=opendir(p.c_str());
  	if(dir){
		while(drnt = readdir(dir)){
			if(drnt->d_name[0] != '.'){
				pch = strtok (drnt->d_name,".");
				while (pch != NULL){
					if(strcmp(pch,"png") == 0){
						path.push_back((string) drnt->d_name);
					}
					pch = strtok (NULL, ".");
				}
			}
		}
 		closedir(dir);
  	} else{
		printf("Não foi possível abrir essa pasta '%s'\n", p.c_str());
		exit(-1);
  	}
	return path;
}

Mat alignImages(Mat _src, Mat _dst){
	Detector face_detector;
	Mat src(_src.rows, _src.cols, CV_8UC3), dst(_dst.rows, _dst.cols, CV_8UC3);
	if(_src.channels() == 1)
		cvtColor(_src, src, CV_GRAY2BGR);
	else
		_src.copyTo(src);
	if(_dst.channels() == 1)
		cvtColor(_dst, dst, CV_GRAY2BGR);
	else
		_dst.copyTo(dst);
	Point2f src_pt[5], dst_pt[5];
	face_detector(src);
	if(face_detector.num_faces() > 0){
		vector<Point> landmarks = face_detector.get_landmarks(0);
		for(int i = 0; i < landmarks.size(); i++){
			circle(src, landmarks[i], 2, Scalar(0,0,255));
			src_pt[i].x = (float)landmarks[i].x;
			src_pt[i].y = (float)landmarks[i].y;
		}
	} else {
		return Mat();
	}
	vector<Point> landmarks;
	landmarks.push_back(Point(41,53));
	landmarks.push_back(Point(84,53));
	landmarks.push_back(Point(61,77));
	landmarks.push_back(Point(46,99));
	landmarks.push_back(Point(79,99));
	for(int i = 0; i < landmarks.size(); i++){
		circle(dst, landmarks[i], 2, Scalar(0,0,255));
		dst_pt[i].x = (float)landmarks[i].x;
		dst_pt[i].y = (float)landmarks[i].y;
	}
	Mat warp_mat = getAffineTransform(src_pt, dst_pt);
	Mat ret(src.rows, src.cols, CV_8UC1);
	warpAffine(src, ret, warp_mat, ret.size());
	if(ret.channels() != 1)
		cvtColor(ret, ret, CV_BGR2GRAY);
	return ret;
}

vector<Mat> imageAligner(Mat mean, vector<Mat> dataset){
	vector<Mat> ret;
	for(Mat m:dataset){
		m = alignImages(m, mean);
		ret.push_back(m);
	}
	return ret;
}

void lbphFacesRecog(string path_input, string path_output, Mat mean){
	vector<string> paths_in = loadDataset(path_input);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_output);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images(paths_in.size()), imagesn(paths_in.size());
	vector<int> labels(paths_in.size());
	for(int i = 0; i < paths_in.size(); i++){
		images[i] = imread(path_input+paths_in[i]+".png", 0);
		labels[i] = stoi(paths_in[i].substr(0,3));
	}

	Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		Mat img = imread(path_output+paths_out[i]+".png", 0);
		int predictedLabel = model->predict(img);
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using LBPH >> Correct: " << correto << ". Wrong: " << errado << endl;
}

void fisherFacesRecog(string path_input, string path_output, Mat mean){
	vector<string> paths_in = loadDataset(path_input);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_output);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images(paths_in.size());
	vector<int> labels(paths_in.size());
	for(int i = 0; i < paths_in.size(); i++){
		images[i] = imread(path_input+paths_in[i]+".png", 0);
		labels[i] = stoi(paths_in[i].substr(0,3));
	}

	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		Mat img = imread(path_output+paths_out[i]+".png", 0);
		int predictedLabel = model->predict(img);
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using FisherFaces >> Correct: " << correto << ". Wrong: " << errado << endl;
}

void eigenFacesRecog(string path_input, string path_output, Mat mean){
	vector<string> paths_in = loadDataset(path_input);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_output);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images(paths_in.size());
	vector<int> labels(paths_in.size());
	for(int i = 0; i < paths_in.size(); i++){
		images[i] = imread(path_input+paths_in[i]+".png", 0);
		labels[i] = stoi(paths_in[i].substr(0,3));
	}

	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		Mat img = imread(path_output+paths_out[i]+".png", 0);
		int predictedLabel = model->predict(img);
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using EigenFaces >> Correct: " << correto << ". Wrong: " << errado << endl;
}

Mat meanFace(string dataset_path){
	Mat ret;
	cout << "Building Mean Face of dataset" << endl;

	vector<string> paths_in = loadDataset(dataset_path);
	vector<Mat> images(paths_in.size());
	vector<int> labels(paths_in.size());
	for(int i = 0; i < paths_in.size(); i++){
		images[i] = imread(dataset_path+paths_in[i]+".png", 0);
		labels[i] = stoi(paths_in[i].substr(0,3));
	}

	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);

	ret = model->getMean();
	ret = norm_0_255(ret.reshape(1, images[0].rows));
	imwrite("meanface.png", ret);
	cout << "Mean face built." << endl;

	return ret;
}

int main(int argc, const char *argv[]) {
	string path_output = "../../../data128_128/inputs/train/";
	string path_input = "../../export/";

	
	Mat mean = imread("meanface.png");
	if(mean.empty())
		mean = meanFace(path_output);
	else
		cout << "Mean face loaded." << endl;

	cout << "Processing Recognition Algorithms" << endl;
	lbphFacesRecog(path_input, path_output, mean);
	eigenFacesRecog(path_input, path_output, mean);
	fisherFacesRecog(path_input, path_output, mean);
	cout << "Facial Recognition Finished" << endl;

	return 0;
}

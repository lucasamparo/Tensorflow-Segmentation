#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>

using namespace cv;
using namespace cv::face;
using namespace std;

static Mat norm_0_255(InputArray _src) {
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

void lbphFacesRecog(string path_input, string path_output){
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

	int height = images[0].rows;
	Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		int predictedLabel = model->predict(imread(path_output+paths_out[i]+".png", 0));
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using LBPH >> Correct: " << correto << ". Wrong: " << errado << endl;
}

void fisherFacesRecog(string path_input, string path_output){
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

	int height = images[0].rows;
	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		int predictedLabel = model->predict(imread(path_output+paths_out[i]+".png", 0));
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using FisherFaces >> Correct: " << correto << ". Wrong: " << errado << endl;
}

void eigenFacesRecog(string path_input, string path_output){
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

	int height = images[0].rows;
	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);

	int correto = 0, errado = 0;
	for(int i = 0; i < paths_out.size(); i++){
		int predictedLabel = model->predict(imread(path_output+paths_out[i]+".png", 0));
		if(predictedLabel == stoi(paths_out[i].substr(0,3)))
			correto++;
		else
			errado++;
	}
	cout << "Matching using EigenFaces >> Correct: " << correto << ". Wrong: " << errado << endl;
}

int main(int argc, const char *argv[]) {
	string path_output = "../../expressional/";
	string path_input = "../../groundtruth/";

	lbphFacesRecog(path_input, path_output);
	eigenFacesRecog(path_input, path_output);
	fisherFacesRecog(path_input, path_output);
	
    return 0;
}

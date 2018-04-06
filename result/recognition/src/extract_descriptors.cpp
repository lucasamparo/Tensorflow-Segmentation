#include <normalize.hpp>
#include <detector.hpp>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <dirent.h>
#include <functional>

#include <opencv2/opencv.hpp>
#include <descriptor.hpp>

using namespace std;
using namespace cv;

struct data{
	Mat desc;
	string name;
};

struct score{
	string name;
	double value;

	bool operator > (const score& str) const{
	        return (value > str.value);
    	}
};

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

void showNMat(Mat img, int n, int m = 0){
	for(int i = 0; i < n; i++){
		if(m > 0){
			for(int j = 0; j < m; j++){
				cout << img.at<double>(i,j) << " ";
			}
			cout << endl;
		} else {
			cout << img.at<double>(i,m) << " ";
		}
	}
	cout << endl;
}

int main(int argc, char const *argv[]) {
	string path_output = "../expressional/new/";
	string path_input = "../groundtruth/new/";
	vector<string> paths_in = loadDataset(path_input);
	vector<int> ids(100);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_output);
	sort(paths_out.begin(), paths_out.end());

	Detector face_detector;
	vector<data> dataset;
	int detected = 0;
	Mat modelo = imread(path_input+"042_000_1557.png");
	for(string p:paths_in){
		Mat input = imread(path_input+p+".png");
		int id = stoi(p.substr(0,3));
		if(ids[id] > 1)
			continue;
		int padding = 64;
		Mat img, model, desc;
		copyMakeBorder( input, img, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		copyMakeBorder( modelo, model, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		face_detector(model);
		Descriptor deepGod;
		ofstream file_desc("descriptors/"+p+".txt");
		if(face_detector.num_faces() == 1) {
			vector<Point> color_landmarks = face_detector.get_landmarks(0);
			cvtColor(img, img, CV_BGR2GRAY);
			Mat normalized_face = normalize(img, color_landmarks).clone();
			desc = deepGod(normalized_face).clone();
			for(int i = 0; i < desc.cols; i++) 
				for(int j = 0; j < desc.rows; j++) 
					file_desc << desc.at<float>(j,i) << endl;
			detected++;
			file_desc.close();	
			data d;
			d.desc = desc;
			d.name = p.substr(0,3);
			ids[id]++;
			dataset.push_back(d);
			cout << "D (" << p << "): ";
			showNMat(desc, 10);
		}
	}
	cout << "Dataset loaded. " << detected << " truly detected on set." << endl;

	vector<int> certo(paths_in.size()), errado(paths_in.size());
	for(string p:paths_out){
		Mat output = imread(path_output+p+".png");
		int padding = 64;
		Mat img, model, desc;
		copyMakeBorder( output, img, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		copyMakeBorder( modelo, model, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		face_detector(model);
		Descriptor deepGod;
		vector<score> scores;
		if(face_detector.num_faces() == 1) {
			vector<Point> color_landmarks = face_detector.get_landmarks(0);
			cvtColor(img, img, CV_BGR2GRAY);
			Mat normalized_face = normalize(img, color_landmarks).clone();
			desc = deepGod(normalized_face).clone();
			double maxScore = 0;
			string maxName;
			for(data d:dataset){
				double s = deepGod.compare(d.desc, desc);
				score s1;
				s1.name = d.name;
				s1.value = s;
				scores.push_back(s1);
			}
			sort(scores.begin(), scores.end(), greater<score>());
			cout << "Sample " << p << ". Best match: " << endl;
			cout << "       " << scores[0].value << " (" << scores[0].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;
			cout << "       " << scores[1].value << " (" << scores[1].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;
			cout << "       " << scores[2].value << " (" << scores[2].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;

			int id = 0;
			for(int i = 0; i < scores.size(); i++){
				for(int j = 0; j < i; j++){
					if(scores[j].name.substr(0,3) == p.substr(0,3)){
						id = j;
						break;
					}
				}
			}
			certo[id]++;
		}
	}

	cout << "#### Resultados ####" << endl;
	int acc = 0;
	for(int i = 0; i < detected; i++){
		acc += certo[i];
		cout << i+1 << "-N >> Certos: " << certo[i] << ". Acumulado: " << acc << endl;
	}

	return 0;
}

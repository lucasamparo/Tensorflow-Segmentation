#include <normalize.hpp>
#include <detector.hpp>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <dirent.h>
#include <functional>
#include <sstream>

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

struct rocitem{
	int falso_pos = 0;
	int verda_pos = 0;
	int falso_neg = 0;
	int verda_neg = 0;
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

	//Expressas
	string path_output = "../expressional/new/";
	string path_input = "../groundtruth/new/";
	//Rede
	//string path_output = "../history/enc-3fc-dec-new/";
	//string path_input = "../gt_rede/new/";
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
	vector<rocitem> roc(int((0.95-0.55)/0.01));

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
			//sort(scores.begin(), scores.end(), greater<score>());
			/*cout << "Sample " << p << ". Best match: " << endl;
			cout << "       " << scores[0].value << " (" << scores[0].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;
			cout << "       " << scores[1].value << " (" << scores[1].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;
			cout << "       " << scores[2].value << " (" << scores[2].name.substr(0,3) << " | " << p.substr(0,3) << ")" << endl;*/

			for(int i = 0; i < scores.size(); i++){
				for(int j = 0; j < i; j++){
					int id = 0;
					for(double r = 0.55; r <= 0.951; r += 0.01){
						if(id > roc.size()-1)
							continue;
						//cout << predicao[j].distance << " " << r << " " << predicao[j].label << " " << path[i].substr(0,3) << endl;
						if(scores[j].value > r){
							if(scores[j].name.substr(0,3) == p.substr(0,3))
								roc[id].falso_neg++;
							else
								roc[id].verda_neg++;
						} else {
							if(scores[j].name.substr(0,3) == p.substr(0,3))
								roc[id].verda_pos++;
							else
								roc[id].falso_pos++;
						}
						id++;
					}
				}
			}
			cout << p << " processado" << endl;
		}
	}

	/*cout << "#### Resultados ####" << endl;
	int acc = 0;
	for(int i = 0; i < detected; i++){
		acc += certo[i];
		cout << i+1 << "-N >> Certos: " << certo[i] << ". Acumulado: " << acc << endl;
	}*/

	stringstream ssx, ssy, c;
	for(int i = 0; i < roc.size(); i++){
		c << i << ",";
		ssx << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].falso_neg) << ",";
		ssy << setprecision(5) << roc[i].verda_pos/float(roc[i].verda_pos + roc[i].verda_neg) << ",";
	}

	cout << c.str() << endl;
	cout << ssx.str() << endl;
	cout << ssy.str() << endl;

	return 0;
}

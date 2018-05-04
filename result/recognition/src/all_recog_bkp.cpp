#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <detector.hpp>
#include <descriptor.hpp>
#include <normalize.hpp>

#include <boost/progress.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <cstring>
#include <utility>
#include <vector>
#include <string>
#include <bits/stdc++.h>
#include <functional>

using namespace cv;
using namespace cv::face;
using namespace std;

struct item{
	string label;
	double distance;
};

struct model{
	string label;
	Mat hog;
};

struct rocitem{
	int falso_pos = 0;
	int verda_pos = 0;
	int falso_neg = 0;
	int verda_neg = 0;
	int genuine = 0;
	int impostor = 0;
};

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

string lbphRecog(string path_gt, string path_img, string label, int result_method = 0, int roc_step = 100, double tr_ini = 0, double tr_fim = 1){
	/*
	Computar Reconhecimento via LBPH, implementação OpenCV
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/

	vector<string> paths_in = loadDataset(path_gt);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_img);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images, imagesn(paths_in.size());
	vector<int> labels;
	vector<int> c_labels(100);
	for(int i = 0; i < paths_in.size(); i++){
		int id = stoi(paths_in[i].substr(0,3));
		if(c_labels[id] > 1)
			continue;
		images.push_back(imread(path_gt+paths_in[i]+".png", 0));
		labels.push_back(id);
		c_labels[id]++;
	}

	Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	vector<int> rankn(labels.size());
	vector<rocitem> roc(roc_step);
	int id = 0;
	
	cout << "LBPH Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	vector<vector< pair<int, double> > > pares_acc;
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < paths_out.size(); i++){
		bool find = false;
		int j = 0, img_id;
		Mat img = imread(path_img+paths_out[i]+".png", 0);
		Ptr<StandardCollector> collector = StandardCollector::create();
		model->predict( img, collector );
		vector< pair<int, double> > pares = collector->getResults();
		//Escrever arquivo
		ofstream gen("matchs/genuine_lbph_"+label+".txt", ios::app), imp("matchs/impostor_lbph_"+label+".txt", ios::app);
		if(gen.is_open() && imp.is_open()){
			for(int x = 0; x < pares.size(); x++){
				if(stoi(paths_out[i].substr(0,3)) == pares[x].first)
					gen << stoi(paths_out[i].substr(0,3)) << " " << pares[x].first << " " << pares[x].second << endl;
				else
					imp << stoi(paths_out[i].substr(0,3)) << " " << pares[x].first << " " << pares[x].second << endl;
			}
			gen.close();
			imp.close();
		}
		

		/*if(result_method == 0){
			for(int x = 0; x < pares.size()-1; x++){
				for(int y = x; y < pares.size(); y++){
					if(pares[x].second > pares[y].second){
						pair<int, double> t = pares[x];
						pares[x] = pares[y];
						pares[y] = t;
					}
				}
			}
			find = false;
			j = 0;
			img_id = stoi(paths_out[i].substr(0,3));
			do{
				if(img_id == pares[j].first){
					find = true;
					rankn[j]++;
				} else {
					j++;
				}
			} while(!find || j == rankn.size());
		} else {
			pares_acc.push_back(pares);
			for(int j = 0; j < pares.size(); j++){
				if(pares[j].second > max_val)
					max_val = pares[j].second;
				if(pares[j].second < min_val)
					min_val = pares[j].second;
			}
		}*/
		++show_progress;
	}
	/*tr_ini = min_val;
	tr_fim = max_val;

	if(result_method == 1){
		double tr = (tr_fim - tr_ini)/roc_step;
		for(vector<pair<int, double> > pares:pares_acc){
			int i = 0;
			for(pair<int, double> p:pares){
				int id = 0;
				for(double r = tr_ini; r <= tr_fim; r += tr){
					if(id > roc.size()-1)
						continue;
					if(p.second > r){
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].falso_neg++;
						else
							roc[id].verda_neg++;
					} else {
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].verda_pos++;
						else
							roc[id].falso_pos++;
					}
					id++;
				}
			}
			i++;
		}
	}

	stringstream ret;
	if (result_method == 0){
		int acc = 0;
		for(int i = 0; i < rankn.size(); i++){
			acc += rankn[i];
			ret << acc;
			if(i < rankn.size() - 1)
				ret << ",";
		}
		return ret.str();
	} else {
		stringstream ssx, ssy;
		for(int i = 0; i < roc.size(); i++){
			ssx << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].verda_neg);
			ssy << setprecision(5) << 1 - roc[i].verda_pos/float(roc[i].verda_pos + roc[i].falso_neg);
			if( i < roc.size() - 1){
				ssx << ",";
				ssy << ",";
			}
		}
		stringstream ret;
		ret << ssx.str() << endl << ssy.str();
		return ret.str();
	}*/
	return "";
}

string eigenRecog(string path_gt, string path_img, int result_method = 0, int roc_step = 100, double tr_ini = 0, double tr_fim = 1){
	/*
	Computar Reconhecimento via EigenFaces, implementação OpenCV
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/
	vector<string> paths_in = loadDataset(path_gt);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_img);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images, imagesn(paths_in.size());
	vector<int> labels;
	vector<int> c_labels(100);
	for(int i = 0; i < paths_in.size(); i++){
		int id = stoi(paths_in[i].substr(0,3));
		if(c_labels[id] > 1)
			continue;
		images.push_back(imread(path_gt+paths_in[i]+".png", 0));
		labels.push_back(id);
		c_labels[id]++;
	}

	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);

	vector<int> rankn(labels.size());
	double tr = (tr_fim - tr_ini)/roc_step;
	vector<rocitem> roc(roc_step);
	int id = 0;
	
	cout << "EigenFaces Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	vector<vector< pair<int, double> > > pares_acc;
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < paths_out.size(); i++){
		bool find = false;
		int j = 0, img_id;
		Mat img = imread(path_img+paths_out[i]+".png", 0);
		Ptr<StandardCollector> collector = StandardCollector::create();
		model->predict( img, collector );
		vector< pair<int, double> > pares = collector->getResults();
		if(result_method == 0){
			for(int x = 0; x < pares.size()-1; x++){
				for(int y = x; y < pares.size(); y++){
					if(pares[x].second > pares[y].second){
						pair<int, double> t = pares[x];
						pares[x] = pares[y];
						pares[y] = t;
					}
				}
			}
			find = false;
			j = 0;
			img_id = stoi(paths_out[i].substr(0,3));
			do{
				if(img_id == pares[j].first){
					find = true;
					rankn[j]++;
				} else {
					j++;
				}
			} while(!find || j == rankn.size());
		} else {
			pares_acc.push_back(pares);
			for(int j = 0; j < pares.size(); j++){
				if(pares[j].second > max_val)
					max_val = pares[j].second;
				if(pares[j].second < min_val)
					min_val = pares[j].second;
			}
		}
		++show_progress;
	}
	tr_ini = min_val;
	tr_fim = max_val;

	if(result_method == 1){
		double tr = (tr_fim - tr_ini)/roc_step;
		for(vector<pair<int, double> > pares:pares_acc){
			int i = 0;
			for(pair<int, double> p:pares){
				int id = 0;
				for(double r = tr_ini; r <= tr_fim; r += tr){
					if(id > roc.size()-1)
						continue;
					if(p.second > r){
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].falso_neg++;
						else
							roc[id].verda_neg++;
					} else {
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].verda_pos++;
						else
							roc[id].falso_pos++;
					}
					id++;
				}
			}
			i++;
		}
	}

	stringstream ret;
	if (result_method == 0){
		int acc = 0;
		for(int i = 0; i < rankn.size(); i++){
			acc += rankn[i];
			ret << acc;
			if(i < rankn.size() - 1)
				ret << ",";
		}
		return ret.str();
	} else {
		stringstream ssx, ssy;
		for(int i = 0; i < roc.size(); i++){
			ssx << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].verda_neg);
			ssy << setprecision(5) << 1 - roc[i].verda_pos/float(roc[i].verda_pos + roc[i].falso_neg);
			if( i < roc.size() - 1){
				ssx << ",";
				ssy << ",";
			}
		}
		stringstream ret;
		ret << ssx.str() << endl << ssy.str();
		return ret.str();
	}
}

string fisherRecog(string path_gt, string path_img, int result_method = 0, int roc_step = 100, double tr_ini = 0, double tr_fim = 1){
	/*
	Computar Reconhecimento via FisherFaces, implementação OpenCV
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/

	vector<string> paths_in = loadDataset(path_gt);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_img);
	sort(paths_out.begin(), paths_out.end());
	vector<Mat> images, imagesn(paths_in.size());
	vector<int> labels;
	vector<int> c_labels(100);
	for(int i = 0; i < paths_in.size(); i++){
		int id = stoi(paths_in[i].substr(0,3));
		if(c_labels[id] > 1)
			continue;
		images.push_back(imread(path_gt+paths_in[i]+".png", 0));
		labels.push_back(id);
		c_labels[id]++;
	}

	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	vector<int> rankn(labels.size());
	double tr = (tr_fim - tr_ini)/roc_step;
	vector<rocitem> roc(roc_step);
	int id = 0;
	
	cout << "FisherFaces Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	vector<vector< pair<int, double> > > pares_acc;
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < paths_out.size(); i++){
		bool find = false;
		int j = 0, img_id;
		Mat img = imread(path_img+paths_out[i]+".png", 0);
		Ptr<StandardCollector> collector = StandardCollector::create();
		model->predict( img, collector );
		vector< pair<int, double> > pares = collector->getResults();
		if(result_method == 0){
			for(int x = 0; x < pares.size()-1; x++){
				for(int y = x; y < pares.size(); y++){
					if(pares[x].second > pares[y].second){
						pair<int, double> t = pares[x];
						pares[x] = pares[y];
						pares[y] = t;
					}
				}
			}
			find = false;
			j = 0;
			img_id = stoi(paths_out[i].substr(0,3));
			do{
				if(img_id == pares[j].first){
					find = true;
					rankn[j]++;
				} else {
					j++;
				}
			} while(!find || j == rankn.size());
		} else {
			pares_acc.push_back(pares);
			for(int j = 0; j < pares.size(); j++){
				if(pares[j].second > max_val)
					max_val = pares[j].second;
				if(pares[j].second < min_val)
					min_val = pares[j].second;
			}
		}
		++show_progress;
	}
	tr_ini = min_val;
	tr_fim = max_val;

	if(result_method == 1){
		double tr = (tr_fim - tr_ini)/roc_step;
		for(vector<pair<int, double> > pares:pares_acc){
			int i = 0;
			for(pair<int, double> p:pares){
				int id = 0;
				for(double r = tr_ini; r <= tr_fim; r += tr){
					if(id > roc.size()-1)
						continue;
					if(p.second > r){
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].falso_neg++;
						else
							roc[id].verda_neg++;
					} else {
						if(p.first == stoi(paths_out[i].substr(0,3)))
							roc[id].verda_pos++;
						else
							roc[id].falso_pos++;
					}
					id++;
				}
			}
			i++;
		}
	}

	stringstream ret;
	if (result_method == 0){
		int acc = 0;
		for(int i = 0; i < rankn.size(); i++){
			acc += rankn[i];
			ret << acc;
			if(i < rankn.size() - 1)
				ret << ",";
		}
		return ret.str();
	} else {
		stringstream ssx, ssy;
		for(int i = 0; i < roc.size(); i++){
			ssx << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].verda_neg);
			ssy << setprecision(5) << 1 - roc[i].verda_pos/float(roc[i].verda_pos + roc[i].falso_neg);
			if( i < roc.size() - 1){
				ssx << ",";
				ssy << ",";
			}
		}
		stringstream ret;
		ret << ssx.str() << endl << ssy.str();
		return ret.str();
	}
}

string cnnRecog(string path_gt, string path_img, int result_method = 0, int roc_step = 100, double tr_ini = 0, double tr_fim = 1){
	/*
	Computar Reconhecimento via EigenFaces, implementação DeepGod
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/
	vector<string> paths_in = loadDataset(path_gt);
	vector<int> ids(100);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_img);
	sort(paths_out.begin(), paths_out.end());

	Detector face_detector;
	vector<data> dataset;
	int detected = 0;
	Mat modelo = imread(path_gt+"042_000_1557.png");
	for(string p:paths_in){
		Mat input = imread(path_gt+p+".png");
		int id = stoi(p.substr(0,3));
		if(ids[id] > 1)
			continue;
		int padding = 64;
		Mat img, model, desc;
		copyMakeBorder( input, img, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		copyMakeBorder( modelo, model, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0,0,0) );
		face_detector(model);
		Descriptor deepGod;
		if(face_detector.num_faces() == 1) {
			vector<Point> color_landmarks = face_detector.get_landmarks(0);
			cvtColor(img, img, CV_BGR2GRAY);
			Mat normalized_face = normalize(img, color_landmarks).clone();
			desc = deepGod(normalized_face).clone();
			detected++;	
			data d;
			d.desc = desc;
			d.name = p.substr(0,3);
			ids[id]++;
			dataset.push_back(d);
		}
	}
	vector<rocitem> roc(roc_step);
	vector<int> rankn(dataset.size());	

	cout << "CNN Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	vector<vector<score> > scores_acc;
	double max_val = 0, min_val = 100000;
	for(string p:paths_out){
		Mat output = imread(path_img+p+".png");
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
			if(result_method == 0){
				sort(scores.begin(), scores.end(), greater<score>());
				bool find = false;
				int j = 0;
				do{
					if(p.substr(0,3) == scores[j].name.substr(0,3)){
						find = true;
						rankn[j]++;
					} else {
						j++;
					}
				} while(!find || j == rankn.size());
			} else {
				scores_acc.push_back(scores);
				for(int i = 0; i < scores.size(); i++){
					if(scores[i].value > max_val)
						max_val = scores[i].value;
					if(scores[i].value < min_val)
						min_val = scores[i].value;	
				}
			}
		}
		++show_progress;
	}
	tr_ini = min_val;
	tr_fim = max_val;
	double tr = (tr_fim - tr_ini)/roc_step;

	if(result_method == 1){
		double tr = (tr_fim - tr_ini)/roc_step;
		for(vector<score> pares:scores_acc){
			int i = 0;
			for(score p:pares){
				int id = 0;
				for(double r = tr_ini; r <= tr_fim; r += tr){
					if(id > roc.size()-1)
						continue;
					if(p.value > r){
						if(p.name.substr(0,3) == paths_out[i].substr(0,3))
							roc[id].falso_neg++;
						else
							roc[id].verda_neg++;
					} else {
						if(p.name.substr(0,3) == paths_out[i].substr(0,3))
							roc[id].verda_pos++;
						else
							roc[id].falso_pos++;
					}
					id++;
				}
			}
			i++;
		}
	}	
	
	stringstream ret;
	if (result_method == 0){
		int acc = 0;
		for(int i = 0; i < rankn.size(); i++){
			acc += rankn[i];
			ret << acc;
			if(i < rankn.size() - 1)
				ret << ",";
		}
		return ret.str();
	} else {
		stringstream ssx, ssy;
		for(int i = 0; i < roc.size(); i++){
			ssx << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].verda_neg);
			ssy << setprecision(5) << 1 - roc[i].verda_pos/float(roc[i].verda_pos + roc[i].falso_neg);
			if( i < roc.size() - 1){
				ssx << ",";
				ssy << ",";
			}
		}
		stringstream ret;
		ret << ssx.str() << endl << ssy.str();
		return ret.str();
	}
}

void computeMagAngle(InputArray src, OutputArray mag, OutputArray ang){
	Mat img = src.getMat();
	img.convertTo(img, CV_32F, 1 / 255.0);

	/// calculate gradients using sobel
	Mat gx, gy;
	Sobel(img, gx, CV_32F, 1, 0, 1);
	Sobel(img, gy, CV_32F, 0, 1, 1);


	/// Calculate gradient magnitude and direction
	Mat magnitude, angle;
	cartToPolar(gx, gy, magnitude, angle, 1);

	mag.assign(magnitude);
	ang.assign(angle);
}

void computeHOG(InputArray mag, InputArray ang, OutputArray dst, int dims, bool isWeighted = true){
	/// init input values
	Mat magMat = mag.getMat();
	Mat angMat = ang.getMat();

	/// validate magnitude and angle dimensions
	if (magMat.rows != angMat.rows || magMat.cols != angMat.cols) {
		return;
	}

	/// get row and col dimensions
	int rows = magMat.rows;
	int cols = magMat.cols;

	/// set up the expected feature dimension, and  
	/// compute the histogram bin length (arc degree) 
	int featureDim = dims;
	float circleDegree = 360.0;
	float binLength = circleDegree / (float)featureDim;
	float halfBin = binLength / 2;

	/// set up the output feature vector
	/// upper limit and median for each bin
	Mat featureVec(1, featureDim, CV_32F);
	featureVec = 0.0;
	vector<float> uplimits(featureDim);
	vector<float> medbins(featureDim);

	for (int i = 0; i < featureDim; i++) {
		uplimits[i] = (2 * i + 1) * halfBin;
		medbins[i] = i * binLength;
	}

	/// begin calculate the feature vector
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			/// get the value of angle and magnitude for 
			/// the current index (i,j)
			float angleVal = angMat.at<float>(i, j);
			float magnitudeVal = magMat.at<float>(i, j);

			/// (this is used to calculate weights)
			float dif = 0.0; /// dfference between the angle and the bin value 
			float prop = 0.0; /// proportion for the value of the current bin 

							  /// value to add for the histogram bin of interest
			float valueToAdd = 0.0;
			/// value to add for the neighbour of histogram bin of interest
			float sideValueToAdd = 0.0;
			/// index for the bin of interest and the neighbour
			int valueIdx = 0;
			int sideIdx = 0;

			/// the first bin (zeroth index) is a little bit tricky 
			/// because its value ranges between below 360 degree and higher 0 degree
			/// we need something more intelligent approach than this
			if (angleVal <= uplimits[0] || angleVal >= uplimits[featureDim - 1]) {

				if (!isWeighted) {
					featureVec.at<float>(0, 0) += magnitudeVal;
				}
				else {
					if (angleVal >= medbins[0] && angleVal <= uplimits[0]) {
						dif = abs(angleVal - medbins[0]);

						valueIdx = 0;
						sideIdx = 1;
					}
					else {
						dif = abs(angleVal - circleDegree);

						valueIdx = 0;
						sideIdx = featureDim - 1;
					}
				}

			}
			/// this is for the second until the last bin
			else {
				for (int k = 0; k < featureDim - 1; k++)
				{
					if (angleVal >= uplimits[k] && angleVal < uplimits[k + 1]) {
						if (!isWeighted) {
							featureVec.at<float>(0, k + 1) += magnitudeVal;
						}
						else {
							dif = abs(angleVal - medbins[k + 1]);
							valueIdx = k + 1;

							if (angleVal >= medbins[k + 1]) {
								sideIdx = (k + 1 == featureDim - 1) ? 0 : k + 2;
							}
							else {
								sideIdx = k;
							}
						}

						break;
					}
				}
			}

			/// add the value proportionally depends of 
			/// how close the angle to the median limits
			if (isWeighted) {
				prop = (binLength - dif) / binLength;
				valueToAdd = prop * magnitudeVal;
				sideValueToAdd = (1.00 - prop) * magnitudeVal;
				featureVec.at<float>(0, valueIdx) += valueToAdd;
				featureVec.at<float>(0, sideIdx) += sideValueToAdd;
			}

			//cout << endl;
			//cout << "-angleVal " << angleVal << " -valueIdx " << valueIdx << " -sideIdx " << sideIdx << endl;
			//cout << "-binLength " << binLength << " -dif " << dif << " -prop " << prop << endl;
			//cout << "binLength - dif " << binLength - dif << " (binLength - dif) / binLength " << (binLength - dif) / binLength << endl;
			//cout << "-> " << featureVec << endl;
		}
	}

	dst.assign(featureVec);
}

vector<model> trainHOGModel(string path){
	vector<string> files = loadDataset(path);
	string pat = path+"/";
	sort(files.begin(), files.end());
	vector<model> ret;
	vector<int> ids(100);

	for (int i = 0; i < files.size(); i++){
		int id = stoi(files[i].substr(0,3));
		if(ids[id] > 1)
			continue;
		string complete = pat+files[i]+".png";
		Mat img;
		Mat in = imread(complete, IMREAD_GRAYSCALE);
		in.convertTo(img, CV_32F);

		Mat hogFeature, mag, ang;
		computeMagAngle(img/255.0, mag, ang);
		computeHOG(mag, ang, hogFeature, 8, true);

		model m;
		m.hog = hogFeature;
		m.label = files[i].substr(0,3);
		ids[id]++;
		ret.push_back(m);
	}
	
	return ret;
}

vector<item> predictHOG(Mat img, vector<model> model){
	//Input HOG	
	Mat hogFeature, mag, ang;
	computeMagAngle(img, mag, ang);
	computeHOG(mag, ang, hogFeature, 8, true);
	//cout << hogFeature << endl;

	//Compute Distances
	vector<item> itens(model.size());
	for(int i = 0; i < model.size(); i++){
		item it;
		it.label = model[i].label;
		it.distance = model[i].hog.dot(hogFeature);
		itens[i] = it;
	}

	for(int i = 0; i < itens.size()-1; i++){
		for(int j = i; j < itens.size(); j++){
			item tmp;
			if(itens[i].distance > itens[j].distance){
				tmp = itens[i];
				itens[i] = itens[j];
				itens[j] = tmp;
			}	
		}
	}
	
	return itens;
}

string hogRecog(string path_gt, string path_img, int result_method = 0, int roc_step = 100,double tr_ini = 0, double tr_fim = 1){
	/*
	Computar Reconhecimento via EigenFaces, implementação com OpenCV
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/
	vector<model> modelo = trainHOGModel(path_gt);	
	vector<string> path = loadDataset(path_img);

	vector<int> rankn(modelo.size());
	vector<rocitem> roc(roc_step);
	double tr_step = (tr_fim - tr_ini)/roc_step;

	double max_dist = 0, min_dist = 1;

	cout << "HoG Computation" << endl;
	boost::progress_display show_progress(path.size());
	int c = 0, gen_size = 0, imp_size = 0;
	vector<vector<item> > pred_acc;
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < path.size(); i++){
		Mat img;
		Mat in = imread(path_img+path[i]+".png");
		bool find = false;
		int j;
		in.convertTo(img, CV_32F);
		vector<item> predicao = predictHOG(img/255.0, modelo);
		if(result_method == 0){
			for(int j = 0; j < predicao.size(); j++){
				if(find)
					continue;
				if(path[i].substr(0,3) == predicao[j].label){
					find = true;
					rankn[j]++;
				}			
			}
		} else {
			gen_size += 2;
			imp_size += 14;
			pred_acc.push_back(predicao);
			if(predicao[predicao.size()-1].distance > max_val)
				max_val = predicao[predicao.size()-1].distance;
			if(predicao[0].distance < min_val)
				min_val = predicao[0].distance;
		}	
		++show_progress;
	}
	tr_ini = min_val;
	tr_fim = max_val;
	double tr = (tr_fim - tr_ini)/roc_step;

	if(result_method == 1){
		double tr = (tr_fim - tr_ini)/roc_step;
		int i = 0;
		for(vector<item> pares:pred_acc){
			for(item p:pares){
				int id = 0;
				string t = (p.label == path[i].substr(0,3)) ? "genuíno" : "impostor";
				for(double r = tr_ini; r <= tr_fim; r += tr){
					if(id > roc.size() - 1)
						continue;
					if(p.label == path[i].substr(0,3)){
						if(p.distance <= r) roc[id].genuine++;
					} else {
						if(p.distance <= r) roc[id].impostor++;
					}
					/*/cout << p.label << " " << path[i].substr(0,3) << " " << t;
					//cout << " " << setprecision(3) << p.distance << " " << setprecision(3) << r << " ";
					//Se for impostor e abaixo do threshold
					if(p.distance <= r && p.label != path[i].substr(0,3)){
						roc[id].impostor += 1;
						//cout << "impostor++ " << roc[id].impostor;
					}
					//Se for genuíno e abaixo do threshold
					if(p.distance <= r && p.label == path[i].substr(0,3)){
						roc[id].genuine += 1;
						//cout << "impostor++ " << roc[id].genuine;
					}
					//cout << endl;
					id++;*/
					/*if(id > roc.size()-1)
						continue;
					if(p.distance >= r){
						if(p.label == path[i].substr(0,3))
							roc[id].falso_neg++;
						else
							roc[id].verda_neg++;
					} else {
						if(p.label == path[i].substr(0,3))
							roc[id].verda_pos++;
						else
							roc[id].falso_pos++;
					}*/
					id++;
				}
			}
			i++;
			//cout << endl;
		}
	}

	stringstream ret;
	if (result_method == 0){
		int acc = 0;
		for(int i = 0; i < rankn.size(); i++){
			acc += rankn[i];
			ret << acc;
			if(i < rankn.size() - 1)
				ret << ",";
		}
		return ret.str();
	} else {
		stringstream frr, far;
		for(int i = 0; i < roc.size(); i++){
			//frr << setprecision(5) << roc[i].falso_pos/float(roc[i].falso_pos + roc[i].verda_neg);
			//far << setprecision(5) << 1 - roc[i].verda_pos/float(roc[i].verda_pos + roc[i].falso_neg);
			frr << setprecision(5) << (double) roc[i].genuine/gen_size;
			far << setprecision(5) << (double) roc[i].impostor/imp_size;
			if( i < roc.size() - 1){
				frr << ",";
				far << ",";
			}
		}
		ret << frr.str() << endl << far.str();
		return ret.str();
	}
}

int main(int argc, char * argv[]){
	string path_neutra = "../history/enc-3fc-dec-new/";
	string path_gt = "../groundtruth/new/";
	string path_gtrede = "../gt_rede/new/";
	string expressas = "../expressional/new/";
	string exp_rede = "../export/";

	string groundtruth, faces;
	string label = "rede";
	if(argv[1] == "rede" || argc < 2){
		groundtruth = path_gtrede;
		faces = path_neutra;
		cout << "Processando Faces Reconstruídas" << endl;
	} else {
		groundtruth = path_gt;
		faces = expressas;
		label = "expressas";
		cout << "Processando Faces Expressas" << endl;
	}

	//Rank N
	int limit_n = 16;
	vector<string> n;
	stringstream count_n;
	for(int i = 0; i < limit_n; i++){
		count_n << i;
		if(i < limit_n - 1)
			count_n << ",";
	}
	//n.push_back(count_n.str());

	//n.push_back(lbphRecog(groundtruth, faces));
	//n.push_back(eigenRecog(groundtruth, faces));
	//n.push_back(fisherRecog(groundtruth, faces));
	//n.push_back(hogRecog(groundtruth, faces));
	//n.push_back(cnnRecog(groundtruth, faces));

	cout << "CSV Rank N" << endl;
	for(string s:n)
		cout << s << endl;

	//Curva ROC
	int limit_r = 100;
	vector<string> r;
	stringstream count_r;
	for(int i = 0; i < limit_r; i++){
		count_r << i;
		if(i < limit_r - 1)
			count_r << ",";
	}
	r.push_back(count_r.str());

	r.push_back(lbphRecog(groundtruth, faces, label, 1));
	//r.push_back(eigenRecog(groundtruth, faces, 1));
	//r.push_back(fisherRecog(groundtruth, faces, 1));
	//r.push_back(hogRecog(groundtruth, faces, 1));
	//r.push_back(cnnRecog(groundtruth, faces, 1));

	cout << "CSV ROC" << endl;
	for(string s:r)
		cout << s << endl;

	return 0;
}

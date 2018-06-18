#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <detector.hpp>
#include <descriptor.hpp>
#include <normalize.hpp>

#include <boost/progress.hpp>

#include <iostream>
#include <cstdio>
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

void lbphRecog(string path_gt, string path_img, string label){
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
	for(int i = 0; i < paths_in.size(); i++){
		int id = stoi(paths_in[i].substr(0,3));
		images.push_back(imread(path_gt+paths_in[i]+".png", 0));
		labels.push_back(id);
	}

	Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);
	
	cout << "LBPH Computation Started" << endl;
	remove(("matchs/genuine_lbph_"+label+".txt").c_str());
	remove(("matchs/impostor_lbph_"+label+".txt").c_str());
	boost::progress_display show_progress(paths_out.size());
	vector<vector< pair<int, double> > > pares_acc;
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
					gen << stoi(paths_out[i].substr(0,3)) << " " << pares[x].first << " " << -1*pares[x].second << endl;
				else
					imp << stoi(paths_out[i].substr(0,3)) << " " << pares[x].first << " " << -1*pares[x].second << endl;
			}
			gen.close();
			imp.close();
		}	
		++show_progress;
	}
	cout << "LBPH Computation Finished" << endl;
}

Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    size_t n = src.size();
    if(n == 0)
        return Mat();
    size_t d = src[0].total();
    Mat data(n, d, rtype);
    for(int i = 0; i < n; i++) {
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        Mat xi = data.row(i);
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

string eigenRecog(string path_gt, string path_img, string label, string treino, string treino2 = ""){
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
	vector<string> path_train = loadDataset(treino);
		if(treino2 != ""){
		vector<string> tmp = loadDataset(treino2);
		path_train.insert(path_train.end(), tmp.begin(), tmp.end());

		//Misturando...
		std::random_device rd;
		std::mt19937 g(rd());
 		std::shuffle(path_train.begin(), path_train.end(), g);
	}
	vector<Mat> images, imagesn(paths_in.size());
	for(int i = 0; i < path_train.size(); i++){
		int id = stoi(path_train[i].substr(0,3));
		images.push_back(imread(treino+path_train[i]+".png", 0));
	}

	//Montando modelo
	Mat data = asRowMatrix(images, CV_32FC1);
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, 20);

	//Projetando as imagens de treino
	vector<int> rankn(paths_in.size());
	vector<Mat> img_treino;
	vector<string> labels;
	for(int i = 0; i < paths_in.size(); i++){
		img_treino.push_back(imread(path_gt+paths_in[i]+".png", 0));
		labels.push_back(paths_in[i].substr(0,3));
	}
	Mat groundtruth = asRowMatrix(img_treino, CV_32FC1);
	Mat gt_projetado = pca.project(groundtruth);

	//Projetando imagens de teste
	cout << "PCA Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < paths_out.size(); i++){
		int j = 0, img_id;
		Mat img = imread(path_img+paths_out[i]+".png", 0);
		Mat img_row;
		if(img.isContinuous()) {
        	    img.reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        } else {
        	    img.clone().reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        }
		Mat projetada = pca.project(img_row);
		vector<item> distancias(gt_projetado.rows);
		for(int j = 0; j < gt_projetado.rows; j++){
			Mat s = gt_projetado.row(j);
			double dist = norm(s,projetada,NORM_L2);
			item it;
			it.label = labels[j];
			it.distance = dist;
			distancias[j] = it;
		}

		ofstream gen("matchs/genuine_eigen_"+label+".txt", ios::app), imp("matchs/impostor_eigen_"+label+".txt", ios::app);
		if(gen.is_open() && imp.is_open()){
			for(int x = 0; x < distancias.size(); x++){
				if(stoi(paths_out[i].substr(0,3)) == distancias[x].label)
					gen << stoi(paths_out[i].substr(0,3)) << " " << distancias[x].label << " " << -1*distancias[x].distance << endl;
				else
					imp << stoi(paths_out[i].substr(0,3)) << " " << distancias[x].label << " " << -1*distancias[x].distance << endl;
			}
			gen.close();
			imp.close();
		}
		++show_progress;
	}

	stringstream ret;
	int acc = 0;
	for(int i = 0; i < rankn.size(); i++){
		acc += rankn[i];
		ret << acc;
		if(i < rankn.size() - 1)
			ret << ",";
	}
	return ret.str();
}

string fisherRecog(string path_gt, string path_img, string label, string treino, string treino2 = "", int result_method = 0){
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
	vector<string> path_train = loadDataset(treino);
	if(treino2 != ""){
		vector<string> tmp = loadDataset(treino2);
		path_train.insert(path_train.end(), tmp.begin(), tmp.end());
	}
	vector<Mat> images, imagesn(paths_in.size());
	vector<int> labels_lda;
	for(int i = 0; i < path_train.size(); i++){
		int id = stoi(path_train[i].substr(0,3));
		images.push_back(imread(treino+path_train[i]+".png", 0));
		labels_lda.push_back(id);

		//Misturando...
		std::random_device rd;
		std::mt19937 g(rd());
 		std::shuffle(path_train.begin(), path_train.end(), g);
	}

	//Montando modelo
	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels_lda);

	Mat vectors = model->getEigenVectors();
	Mat values = model->getEigenValues();
	Mat mean = model->getMean();
	Mat fishervectors, fishervalues, fishermean;
	vectors.convertTo(fishervectors, CV_32FC1);
	values.convertTo(fishervalues, CV_32FC1);
	mean.convertTo(fishermean, CV_32FC1);

	//Projecao calculada por Proj = (Input - mean) * eigenvectors
	//Montando Conjunto de treino
	vector<Mat> images_treino;
	vector<string> labels_treino;
	for(int i = 0; i < paths_in.size(); i++){
		Mat img = imread(path_gt+paths_in[i]+".png", 0);
		Mat img_row;
		if(img.isContinuous()) {
        	    img.reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        } else {
        	    img.clone().reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        }
		Mat projecao = (img_row - fishermean) * fishervectors;
		images_treino.push_back(projecao);
		labels_treino.push_back(paths_in[i].substr(0,3));
	}

	vector<int> rankn(labels_treino.size());
	int id = 0;

	cout << "FisherFaces Computation" << endl;
	boost::progress_display show_progress(paths_out.size());
	double max_val = 0, min_val = 100000;
	for(int i = 0; i < paths_out.size(); i++){
		Mat img = imread(path_img+paths_out[i]+".png", 0);
		Mat img_row;
		if(img.isContinuous()) {
        	    img.reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        } else {
        	    img.clone().reshape(1, 1).convertTo(img_row, CV_32FC1, 1, 0);
	        }
		Mat projecao = (img_row - fishermean) * fishervectors;

		vector<item> distancias(images_treino.size());
		for(int j = 0; j < images_treino.size(); j++){
			Mat s = images_treino[j];
			double dist = norm(s,projecao,NORM_L2);
			item it;
			it.label = labels_treino[j];
			it.distance = dist;
			distancias[j] = it;
		}

		ofstream gen("matchs/genuine_eigen_"+label+".txt", ios::app), imp("matchs/impostor_eigen_"+label+".txt", ios::app);
		if(gen.is_open() && imp.is_open()){
			for(int x = 0; x < distancias.size(); x++){
				if(stoi(paths_out[i].substr(0,3)) == distancias[x].label)
					gen << stoi(paths_out[i].substr(0,3)) << " " << distancias[x].label << " " << -1*distancias[x].distance << endl;
				else
					imp << stoi(paths_out[i].substr(0,3)) << " " << distancias[x].label << " " << -1*distancias[x].distance << endl;
			}
			gen.close();
			imp.close();
		}

		++show_progress;
	}

	stringstream ret;
	int acc = 0;
	for(int i = 0; i < rankn.size(); i++){
		acc += rankn[i];
		ret << acc;
		if(i < rankn.size() - 1)
			ret << ",";
	}
	return ret.str();
}

void cnnRecog(string path_gt, string path_img, string label){
	/*
	Computar Reconhecimento via EigenFaces, implementação DeepGod
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/
	vector<string> paths_in = loadDataset(path_gt);
	sort(paths_in.begin(), paths_in.end());
	vector<string> paths_out = loadDataset(path_img);
	sort(paths_out.begin(), paths_out.end());

	Detector face_detector;
	vector<data> dataset;
	int detected = 0;
	Mat modelo = imread(path_gt+"100_000.png");
	for(string p:paths_in){
		Mat input = imread(path_gt+p+".png");
		int id = stoi(p.substr(0,3));
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
			dataset.push_back(d);
		}
	}

	cout << "CNN Computation" << endl;
	remove(("matchs/genuine_cnn_"+label+".txt").c_str());
	remove(("matchs/impostor_cnn_"+label+".txt").c_str());
	boost::progress_display show_progress(paths_out.size());
	vector<vector<score> > scores_acc;
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
			//Escrever arquivo
			ofstream gen("matchs/genuine_cnn_"+label+".txt", ios::app), imp("matchs/impostor_cnn_"+label+".txt", ios::app);
			if(gen.is_open() && imp.is_open()){
				for(int x = 0; x < scores.size(); x++){
					if(p.substr(0,3) == scores[x].name.substr(0,3))
						gen << stoi(p.substr(0,3)) << " " << stoi(scores[x].name.substr(0,3)) << " " << scores[x].value << endl;
					else
						imp << stoi(p.substr(0,3)) << " " << stoi(scores[x].name.substr(0,3)) << " " << scores[x].value << endl;
				}
				gen.close();
				imp.close();
			}
		}
		++show_progress;
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

	for (int i = 0; i < files.size(); i++){
		int id = stoi(files[i].substr(0,3));
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

void hogRecog(string path_gt, string path_img, string label){
	/*
	Computar Reconhecimento via EigenFaces, implementação com OpenCV
	Input: Path_gt = imagens de treino; Path_img = imagens de teste;
		Opcional: result_method = 0 (Rank N), 1 (ROC); tr_ini = inicio do threshold; tr_fim = final do threshold;
	Output: Resultados do reconhecimento, para plotar. Rank N ou Curva ROC
	*/
	vector<model> modelo = trainHOGModel(path_gt);	
	vector<string> path = loadDataset(path_img);

	cout << "HoG Computation" << endl;
	remove(("matchs/genuine_hog_"+label+".txt").c_str());
	remove(("matchs/impostor_hog_"+label+".txt").c_str());
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
		//Escrever arquivo
		ofstream gen("matchs/genuine_hog_"+label+".txt", ios::app), imp("matchs/impostor_hog_"+label+".txt", ios::app);
		if(gen.is_open() && imp.is_open()){
			for(int x = 0; x < predicao.size(); x++){
				if(path[i].substr(0,3) == predicao[x].label)
					gen << stoi(path[i].substr(0,3)) << " " << stoi(predicao[x].label) << " " << predicao[x].distance << endl;
				else
					imp << stoi(path[i].substr(0,3)) << " " << stoi(predicao[x].label) << " " << predicao[x].distance << endl;
			}
			gen.close();
			imp.close();
		}	
		++show_progress;
	}
}

int main(int argc, char * argv[]){
	string path_neutra = "../history/with_recog/";
	string path_gt = "../groundtruth/with_recog/";
	string path_gtrede = "../gt_rede/with_recog/";
	string expressas = "../expressional/with_recog/";
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

	lbphRecog(groundtruth, faces, label);
	eigenRecog(groundtruth, faces, label);
	fisherRecog(groundtruth, faces, label);
	hogRecog(groundtruth, faces, label);
	cnnRecog(groundtruth, faces, label);

	return 0;
}

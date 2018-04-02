#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <detector.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <cstring>

using namespace cv;
using namespace cv::face;
using namespace std;

void writeCSV(string filename, vector<auto> data, vector<string> labels){
	ofstream myfile;
	myfile.open(filename.c_str());
	for(int i = 0; i < labels.size(); i++){
		myfile << labels[i];
		if(i < labels.size()-1)
			myfile << ",";
	}
	myfile << endl;
	int c = 0;
	for(int j = 0; j < data[0].size(); j++){
		for(int i = 0; i < labels.size(); i++){
			myfile << data[i][j];
			if(i < labels.size()-1)
				myfile << ",";
		}
		myfile << endl;
	}

	myfile.close();
	cout << filename << " salvo com sucesso." << endl;
}

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

Mat variation(Mat batch){
	Mat ret(batch.size(), CV_8U);
	for(int i = 0; i < batch.rows; i++){
		for(int j = 0; j < batch.cols; j++){
			int p = (int)batch.at<uchar>(i,j);
			int q = (int)batch.at<uchar>(i,j+1);
			int d = 0;
			if(p > 50 && q > 50)
				d = abs(p - q);
			ret.at<uchar>(i,j) = (uchar)d;
		}
	}
	return ret;
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

void calculeCurvatures(Mat batch, double &K, double &H){
	cvtColor(batch,batch,CV_BGR2GRAY);
	Mat Dx, Dy;
	Sobel(batch, Dx, CV_64FC1, 1, 0, 3);
	Sobel(batch, Dy, CV_64FC1, 0, 1, 3);

	Mat Dxx, Dxy;
	Sobel(Dx, Dxx, CV_64FC1, 1, 0, 3);
	Sobel(Dx, Dxy, CV_64FC1, 0, 1, 3);

	Mat Dyy;
	Sobel(Dy, Dyy, CV_64FC1, 0, 1, 3);

	Scalar h = mean(Dxx + Dyy);
	Scalar k = mean((Dxx * Dyy) - (Dxy * Dxy));
	H = h[0];
	K = k[0];
}

void calculeNose(Mat img, Point &nosetip, Point &cornerl, Point &cornerr){
	vector<int> max(img.rows), mean(img.rows), diff(img.rows), x(img.rows);
	
	for(int i = 0; i < img.rows; i++){
		x[i] = i;
		mean[i] = 0;
		max[i] = 0;
		for(int j = 0; j < img.cols; j++){
			if(max[i] < img.at<Vec3b>(i,j).val[0]){
				max[i] = img.at<Vec3b>(i,j).val[0];
			}
			mean[i] += img.at<Vec3b>(i,j).val[0];
		}
		mean[i] /= img.cols;
	}
	
	for(int i = 0; i < mean.size(); i++){
		diff[i] = max[i] - mean[i];
	}

	//Exportar para fazer grafico
	vector<vector<int> > data;
	data.push_back(x);
	data.push_back(max);
	data.push_back(mean);
	data.push_back(diff);
	vector<string> labels(data.size());
	labels = {"x","max","mean","diff"};
	writeCSV("dataY.csv", data, labels);

	int step = diff.size()/3;
	vector<int> sum(3);
	int maxSum = 0, maxInt = 0;
	for(int i = 0; i < 3; i++){
		sum[i] = 0;
		for(int j = step*i; j < step*(i+1); j++){
			sum[i] += diff[j];
		}
		if(maxSum < sum[i]){
			maxSum = sum[i];
			maxInt = i;
		}
	}

	int maxId = 0, maxV = 0;
	for(int i = step*maxInt; i < step*(maxInt+1); i++){
		if(diff[i] > maxV){
			maxV = diff[i];
			maxId = i;
		}
	}

	Mat batch, Dx;
	img.row(maxId).copyTo(batch);
	Dx = variation(batch);
	int maxLv = 0, maxLid = 0, maxRv = 0, maxRid = 0;

	for(int i = 0; i < Dx.cols; i++){
		int d = (int)Dx.at<uchar>(0,i);
		if(i > Dx.cols/2){
			//Direita
			if(maxRv < d){
				maxRv = d;
				maxRid = i;
			}
		} else {
			//Esquerda
			if(maxLv < d){
				maxLv = d;
				maxLid = i;
			}
		}
	}

	int maxXid = 0, maxXv = 0;
	for(int i = maxLid; i < (maxRid+1); i++){
		int v = (int)batch.at<uchar>(0,i);
		if(maxXv < v){
			maxXv = v;
			maxXid = i;
		}
	}

	nosetip.x = maxXid;
	nosetip.y = maxId;

	cornerl.x = maxLid;
	cornerl.y = maxId;

	cornerr.x = maxRid;
	cornerr.y = maxId;
}

void calculeEye(Mat pit,Point &eyel,Point &eyer){
	vector<double> rowc(pit.rows);
	int maxY = 0;
	double maxYv = 0;
	//Descobrindo Y
	for(int i = 0; i < pit.rows; i++){
		rowc[i] = 0;
		for(int j = 0; j < pit.cols; j++){
			if(255 == (int) pit.at<Vec3b>(i,j).val[0]){
				rowc[i] += 1.0;
			}
		}
		rowc[i] /= pit.cols;
		if(rowc[i] > maxYv){
			maxYv = rowc[i];
			maxY = i;
		}
	}

	eyel.y = maxY;
	eyer.y = maxY;

	//Descobrindo X
	vector<double> colc(pit.cols);
	int maxXl = 0, maxXr;
	double maxXlv = 0, maxXrv = 0;
	for(int i = 3; i < pit.cols-3; i++){
		int c = 0;
		for(int j = i-3; j < i+3; j++){
			if((int)pit.at<Vec3b>(maxY,j).val[0] == 255){
				c++;
			}
		}
		colc[i] = c/7.0;
	}

	int idm1 = 0, idm2 = 0;
	double vm1 = 0, vm2 = 0;
	for(int i = 0; i < colc.size()/2; i++){
		if(colc[i] > vm1){
			vm1 = colc[i];
			idm1 = i;
		}
	}
	for(int i = colc.size()/2; i < colc.size(); i++){
		if(colc[i] > vm2){
			vm2 = colc[i];
			idm2 = i;
		}
	}
	
	if(idm1 > pit.cols/2){
		maxXl = idm2;
		maxXr = idm1;
	} else {
		maxXl = idm1;
		maxXr = idm2;
	}

	eyel.x = maxXl;
	eyer.x = maxXr;
}

void computeCurvatures(Mat src, Mat &pit, Mat &peak){
	int WINDOW = 7;
	peak.setTo(Vec3b(0,0,0));
	pit.setTo(Vec3b(0,0,0));
	for(int i = 0; i < src.rows-WINDOW; i++){
		for(int j = 0; j < src.cols-WINDOW; j++){
			Rect roi;
			roi.x = j;
			roi.y = i;
			roi.width = WINDOW;
			roi.height = WINDOW;
			Mat batch = src(roi);
			double K, H;
			calculeCurvatures(batch, K, H);
			//Pit = azul; Peak = Vermelho
			if(K > 0 && H > 0){
				//Achou Pit
				pit.at<Vec3b>(i,j) = Vec3b(255,0,0);
			}
			if(K > 0 && H < 0){
				//Achou Peak
				peak.at<Vec3b>(i,j) = Vec3b(0,0,255);
			}
		}
	}
}

Mat alignImages(Mat _src, Mat _dst){
	//Implementar o detector de Maurício
	Mat peak(_src.size(), CV_32FC3), pit(_src.size(), CV_32FC3), mask, src, dst;
	bilateralFilter(_src, src, 5, 150, 5);
	threshold(src, mask, 50, 255,0);
	//Detectando as features da source
	cout << "Computando curvaturas" << endl;
	computeCurvatures(src,pit,peak);
	cout << "Curvaturas computadas" << endl;
	
	mask.convertTo(mask,CV_32FC3);
	cout << pit.type() << " " << peak.type() << " " << mask.type() << endl;
	bitwise_and(pit,mask,pit);
	bitwise_and(peak,mask,peak);
	imshow("pit", pit);
	imshow("peak", peak);

	Point nosetip, cornerl, cornerr;
	cout << "Calculando nosetip" << endl;
	calculeNose(src, nosetip, cornerl, cornerr);
	circle(src, nosetip, 3, Scalar(255,0,0), -1);
	circle(src, cornerl, 3, Scalar(255,0,0), -1);
	circle(src, cornerr, 3, Scalar(255,0,0), -1);
	Point eyel, eyer;
	cout << "Calculando Olhos" << endl;
	calculeEye(pit, eyel, eyer);
	circle(src, eyel, 3, Scalar(255,0,0), -1);
	circle(src, eyer, 3, Scalar(255,0,0), -1);
	cout << nosetip << " " << cornerl << " " << cornerr << " " << eyel << " " << eyer << endl;
	vector<Mat> mats = {src,pit,peak,mask};
	imshow("Nosetip", src);
	waitKey(0);

	return src;
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
	string path_input = "../groundtruth/renorm/";
	string path_output = "../expressional/renorm/";

	
	Mat mean = imread("meanface.png");
	if(mean.empty())
		mean = meanFace(path_output);
	else
		cout << "Mean face loaded." << endl;

	/*vector<string> p = loadDataset(path_output);

	for(int i = 0; i < p.size(); i++){
		Mat test = imread(path_output+p[i]+".png");
		Mat aligned = alignImages(test, mean);	
	}*/

	//imshow("Teste", test);
	//imshow("Alinhada", aligned);
	//imshow("Media", mean);
	//waitKey(0);

	cout << "Processing Recognition Algorithms" << endl;
	lbphFacesRecog(path_input, path_output, mean);
	eigenFacesRecog(path_input, path_output, mean);
	fisherFacesRecog(path_input, path_output, mean);
	cout << "Facial Recognition Finished" << endl;

	return 0;
}

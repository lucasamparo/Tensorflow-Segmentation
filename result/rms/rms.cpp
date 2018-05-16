#include <iostream>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace cv;
using namespace std;

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

Scalar getMSSIM( const Mat& i1, const Mat& i2){
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

double computeRMS(Mat a, Mat b){
	double acc = 0;
	double rms = 0;
	for(int i = 0; i < a.rows; i++){
		for(int j = 0; j < a.cols; j++){
			int v = abs((int)(a.at<uchar>(i,j) - b.at<uchar>(i,j)));
			acc += pow(v/255.0, 2);
		}
	}
	rms = sqrt(acc/(a.rows*a.cols));
	return rms;
}

void processRMS(string path_gt, string path_exp, string label){
	vector<string> express = loadDataset(path_exp);
	sort(express.begin(), express.end());
	vector<double> rms;

	for(int i = 0; i < express.size(); i++){
		string p = express[i];
		Mat a = imread(path_gt+"/"+p+".png");
		Mat b = imread(path_exp+"/"+express[i]+".png");
		double val = computeRMS(a,b);
		rms.push_back(val);
	}

	double acc = 0;
	double max_val = 0, min_val = 1000;
	for(int i = 0; i < rms.size(); i++){
		acc += rms[i];
		max_val = max(max_val,rms[i]);
		min_val = min(min_val,rms[i]);
	}

	double mean = acc/rms.size();
	acc = 0;
	for(int i = 0; i < rms.size(); i++){
		acc += pow(rms[i]-mean,2);
	}
	double std_dev = acc/rms.size();

	cout << "RMS ("<< label << "): "<< endl;
	cout << "Mean: " << mean << " | Std. Deviation: " << std_dev << " | Max: " << max_val << " | Min: " << min_val << endl;
}

void processMSSIM(string path_gt, string path_exp, string label){
	vector<string> express = loadDataset(path_exp);
	sort(express.begin(), express.end());
	vector<double> rms;

	for(int i = 0; i < express.size(); i++){
		string p = express[i];
		Mat a = imread(path_gt+"/"+p+".png");
		Mat b = imread(path_exp+"/"+express[i]+".png");
		double val = getMSSIM(a,b)[0];
		rms.push_back(val);
	}

	double acc = 0;
	double max_val = 0, min_val = 1000;
	for(int i = 0; i < rms.size(); i++){
		acc += rms[i];
		max_val = max(max_val,rms[i]);
		min_val = min(min_val,rms[i]);
	}

	double mean = acc/rms.size();
	acc = 0;
	for(int i = 0; i < rms.size(); i++){
		acc += pow(rms[i]-mean,2);
	}
	double std_dev = acc/rms.size();

	cout << "MSSIM ("<< label << "): "<< endl;
	cout << "Mean: " << mean << " | Std. Deviation: " << std_dev << " | Max: " << max_val << " | Min: " << min_val << endl;
}

int main(){	
	processRMS("../gt_rede/new/", "../history/enc-3fc-dec-new/", "Faces da Rede");
	processRMS("../groundtruth/new/", "../expressional/new/", "Faces sem Rede");

	processMSSIM("../gt_rede/new/", "../history/enc-3fc-dec-new/", "Faces da Rede");
	processMSSIM("../groundtruth/new/", "../expressional/new/", "Faces sem Rede");
	
	processRMS("../groundtruth/new", "../history/enc-3fc-dec-new", "Remoção da Rede x Neutras Sem rede");
	processMSSIM("../groundtruth/new", "../history/enc-3fc-dec-new", "Remoção da Rede x Neutras Sem rede");

	return 0;	
}

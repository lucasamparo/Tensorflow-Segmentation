#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define K_DROP 15000
#define SQRT_2 1.4142135623730951
#define BASE
#define ROC 0
#define COLOR_ONLY 1
#define METHOD_OLD 1
#define METHOD_NEW 0
#define METHOD_ERF 0
#define METHOD_TANH 0
#define USE_MEDIAN 0
#define MEDIAN_WINDOW 20

#define LOWER_Q 0.723897//0.6376 //0.723897
#define UPPER_Q 0.851387

#ifdef LFW
#define INTRA_MEAN 0.6268834203
#define INTRA_STDDEV 0.1274434326
#define INTER_MEAN 0.0093609024
#define INTER_STDDEV 0.0929235227
#endif
#ifdef BASE
#define INTRA_MEAN 98.1926//0.725852
#define INTRA_STDDEV 19.1682//0.126884
#define INTER_MEAN 181.286//0.0841153
#define INTER_STDDEV 22.7111//0.113743
#endif
#ifdef MULTI
#define INTRA_MEAN 0.776487
#define INTRA_STDDEV 0.10884
#define INTER_MEAN 0.123311
#define INTER_STDDEV 0.119197
#endif
#ifdef MULTI_NEW
#define INTRA_MEAN 0.845324
#define INTRA_STDDEV 0.123111
#define INTER_MEAN 0.0726829
#define INTER_STDDEV 0.134153
#endif
const double MEAN = (INTRA_MEAN + INTER_MEAN) / 2;

double pSafe = 1.0, pNotSafe = 0.0;
long now, lastObservation;
double lastKsafe = 1.0;

//cosside distance of two descriptors
// double compare(const Mat & a, const Mat & b) {
//     return a.dot(b) / cv::norm(a) / cv::norm(b);
// }
//L2
double compare(const Mat & v1, const Mat & v2) {
	assert(v1.size() == v2.size());
	return norm(v1,v2,NORM_L2);
}

//load descriptor as math
void read_contents(const string & path, Mat & v, long & timeStamp, bool & ausence) {
    ifstream file(path);
    if (file.is_open()) {
        float d;
        int count = 0;
        file >> timeStamp;
        file >> d;
        if(d == 1) {
        	ausence = true;
        	return;
        }
        while (file >> d) {
            v.at<float>(0,count) = d;
            count++;
        }
        ausence = false;
    }
    else {
        cerr << "Unable to read file " << path << endl;
        return;
    }
}

// functions to calculate PSafe
void update_psafe(const bool & genuiene) {
    const double dtime = now - lastObservation;
    if(ROC)
    	cout << "1 " << genuiene << " " << std::pow(2.0, -dtime/K_DROP) * pSafe / (pSafe + pNotSafe) << endl;
    else
    	cout << std::pow(2.0, -dtime/K_DROP) * pSafe / (pSafe + pNotSafe) << endl;
}

void calculate_psafe(const double distance, const bool & genuiene) {
    const double dtime = now - lastObservation;
    const double time_decay = std::pow(2.0, -dtime / K_DROP);
    double ksafe, knotsafe;
    if(METHOD_NEW) {
    	if(distance >= LOWER_Q) 
    		ksafe = 0.9999;
    	else
    		ksafe = 0.0001;
    	knotsafe = 1 - ksafe;
	    // if(distance <= INTRA_MEAN - (2*INTRA_STDDEV)) {
	    // 	ksafe = 0.0001;
	    // }
	    // else {
	    // 	ksafe = (distance - INTRA_MEAN + (2*INTRA_STDDEV)) / (INTRA_STDDEV);
	    // 	ksafe = std::erf(ksafe);
	    // }
	    // knotsafe = 1 - ksafe;
	    // else if(distance > INTRA_MEAN - INTRA_STDDEV && distance <= INTRA_MEAN + INTRA_STDDEV) {
	    // 	// ksafe = (2 * (distance - (INTRA_MEAN - INTRA_STDDEV))) / INTRA_STDDEV;
	    // 	ksafe = (distance - INTRA_MEAN + INTRA_STDDEV) / (2 * INTRA_STDDEV);
	    // 	ksafe = std::erf(ksafe);
	    // }
	    // else {
	    // 	ksafe = 1.0;
	    // }
	    // if(distance < INTER_MEAN + INTER_STDDEV) {
	    // 	knotsafe = 1.0;
	    // }
	    // else if(distance <= MEAN) {
	    // 	// knotsafe = (2 * (distance - MEAN)) / (-INTER_MEAN - INTER_STDDEV);
	    // 	knotsafe = (distance - MEAN) / (-INTER_MEAN - INTER_STDDEV + MEAN);
	    // 	knotsafe = 1 - std::erf(knotsafe);
	    // 	knotsafe = (knotsafe / 2) + 0.5;
	    // 	// knotsafe = 1 - std::erf(knotsafe);
	    // }
	    // else if(distance > MEAN && distance < INTRA_MEAN) {
	    // 	// knotsafe = (2 * (distance - MEAN)) / (INTRA_MEAN - INTRA_STDDEV - MEAN);
	    // 	knotsafe = (distance - INTRA_MEAN) / (-MEAN + INTRA_MEAN);
	    // 	knotsafe = 1 - std::erf(knotsafe);
	    // 	knotsafe = knotsafe/2;
	    // 	// knotsafe = 1 - std::erf(knotsafe);
	    // }
	    // else {
	    // 	knotsafe = 0.0001;
	    // }
    }
    if(METHOD_OLD) {
    	// ksafe = 1.0 - (1.0 + std::erf((INTRA_MEAN - distance) / (INTRA_STDDEV * SQRT_2))) / 2;
    	// knotsafe = (1.0 + std::erf((INTER_MEAN - distance) / (INTER_STDDEV * SQRT_2))) / 2;
    	ksafe = 1.0 - (1.0 + std::erf((distance - INTRA_MEAN) / (INTRA_STDDEV * SQRT_2))) / 2;
    	knotsafe = (1.0 + std::erf((distance - INTER_MEAN) / (INTER_STDDEV * SQRT_2))) / 2;
    }
    if(METHOD_ERF) {
    	ksafe = 1.0 - (1.0 + std::erf((INTRA_MEAN - distance) / (INTRA_STDDEV * SQRT_2))) / 2;
    	knotsafe = 1 - ksafe;
    }
    if(METHOD_TANH) {
    	ksafe = 1.0 - (1.0 + std::tanh((INTRA_MEAN - distance) / (INTRA_STDDEV * 2))) / 2;
    	knotsafe = 1 - ksafe;
    }
    // cout << distance << endl;
    pSafe = ksafe + time_decay * pSafe;
    pNotSafe = knotsafe + time_decay * pNotSafe;
    lastKsafe = ksafe;
    update_psafe(genuiene);
}


double median(vector<double> scores) {
	sort(scores.begin(), scores.end());
	if (MEDIAN_WINDOW  % 2 == 0) 
      return (scores[(MEDIAN_WINDOW / 2) - 1] + scores[MEDIAN_WINDOW / 2]) / 2;
	else 
	  return scores[MEDIAN_WINDOW / 2];
}

double mean(vector<double> scores) {
	double sum = 0;
	for(double x : scores)
		sum += 	x;
	return sum / scores.size();
}

int main(int argc, char const *argv[])
{
	if (argc <= 1) {
        cout << "Insufficient arguments. Usage: pass at least one csv to descriptors" << endl;
        return 1;
    }
    if((argc-1) % 2 != 0) {
    	cout << "Invalid arguments. Usage: pass color and ir csvs to descriptors" << endl;
        return 1;
    }
	vector<string> args(argv + 1, argv + argc);
	ifstream fileLoginColor(args[0]);
	ifstream fileLoginIr(args[1]);
	string fileHandlerColor, fileHandlerIr;
	bool ausence = false;
	// login user with the first descriptor
	fileLoginColor >> fileHandlerColor;
	fileLoginIr >> fileHandlerIr;
	Mat logedDescriptorColor = Mat(1,256, CV_32F);
	Mat logedDescriptorIr = Mat(1,256, CV_32F);
	read_contents(fileHandlerColor, logedDescriptorColor, lastObservation, ausence);
	read_contents(fileHandlerIr, logedDescriptorIr, lastObservation, ausence);
	Mat logedUser;
	vector<double> window;
	if(COLOR_ONLY)
		logedUser = logedDescriptorColor;
	else
		logedUser = logedDescriptorColor + logedDescriptorIr;
	// auth continuous for allowed user
	while(fileLoginColor >> fileHandlerColor && fileLoginIr >> fileHandlerIr) {
		Mat observationColor = Mat(1,256, CV_32F);
		Mat observationIr = Mat(1,256, CV_32F);
		read_contents(fileHandlerColor, observationColor, now, ausence);
		read_contents(fileHandlerIr, observationIr, now, ausence);
		if(!ausence) {
			if(USE_MEDIAN) {
				if(window.size() < MEDIAN_WINDOW) {
					if(COLOR_ONLY) 
						window.push_back(compare(observationColor,logedUser));
					else 
						window.push_back(compare(observationColor+observationIr,logedUser));
					if(window.size() == MEDIAN_WINDOW) {
						// cout << median(window) << endl;
						calculate_psafe(mean(window), true);
						window.clear();
					}
					lastObservation = now;
				}
			}
			else {
				if(COLOR_ONLY)
					calculate_psafe(compare(observationColor,logedUser), true);
				else
					calculate_psafe(compare(observationColor+observationIr,logedUser), true);
				lastObservation = now;
			}
		}
		else  {
			update_psafe(true);
		}
		
	}
	if(!ROC)
		cout << "-----------------" << endl;
	// save the pSafe at the end of allowed user
	window.clear();
	double copyPSafe = pSafe, copyPNotSafe = pNotSafe;
	for(int c = 2; c < args.size(); c+=2) {
		ifstream fileLoginColor(args[c]);
		ifstream fileLoginIr(args[c+1]);
		// ignore the first frame for timestamp correction
		fileLoginColor >> fileHandlerColor;
		fileLoginIr >> fileHandlerIr;
		Mat observationColor = Mat(1,256, CV_32F);
		Mat observationIr = Mat(1,256, CV_32F);
		read_contents(fileHandlerColor, observationColor, lastObservation, ausence);
		read_contents(fileHandlerIr, observationIr, lastObservation, ausence);
		//	auth cont stuff
		while(fileLoginColor >> fileHandlerColor && fileLoginIr >> fileHandlerIr) {
			Mat observationColor = Mat(1,256, CV_32F);
			Mat observationIr = Mat(1,256, CV_32F);
			read_contents(fileHandlerColor, observationColor, now, ausence);
			read_contents(fileHandlerIr, observationIr, now, ausence);
			if(!ausence) {
				if(USE_MEDIAN) {
					if(window.size() < MEDIAN_WINDOW) {
						if(COLOR_ONLY) 
							window.push_back(compare(observationColor,logedUser));
						else 
							window.push_back(compare(observationColor+observationIr,logedUser));
						if(window.size() == MEDIAN_WINDOW) {
							// cout << median(window) << endl;
							calculate_psafe(mean(window), false);
							window.clear();
						}
						lastObservation = now;
					}
				}
				else {
					if(COLOR_ONLY)
						calculate_psafe(compare(observationColor,logedUser), false);
					else
						calculate_psafe(compare(observationColor+observationIr,logedUser), false);
					lastObservation = now;
				}
			}
			else {
				update_psafe(false);
			}
		}
		window.clear();
		pSafe = copyPSafe;
		pNotSafe = copyPNotSafe;
		if(!ROC)
			cout << "-----------------" << endl;
	}
	return 0;
}
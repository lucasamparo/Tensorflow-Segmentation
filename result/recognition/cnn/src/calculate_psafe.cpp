#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define LAST_X 32
#define COMPARE_LOGIN 0


//cosside distance of two descriptors
double compare(const Mat & a, const Mat & b) {
    return a.dot(b) / cv::norm(a) / cv::norm(b);
}
//L2
// double compare(const Mat & v1, const Mat & v2) {
// 	assert(v1.size() == v2.size());
// 	return norm(v1,v2,NORM_L2);
// }

//load descriptor as mat
void read_contents(const string & path, Mat & v, bool & ausence) {
    ifstream file(path);
    if (file.is_open()) {
        float d;
        int count = 0;
        long timeStamp = 0;
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
	string fileHandlerColor;
	bool ausence = false;
	// login user with the first descriptor
	fileLoginColor >> fileHandlerColor;
	Mat logedDescriptorColor = Mat(1,256, CV_32F);
	read_contents(fileHandlerColor, logedDescriptorColor, ausence);
	Mat logedUser;
	logedUser = logedDescriptorColor;
	deque<Mat> history;
	// auth continuous for allowed user
	while(fileLoginColor >> fileHandlerColor) {
		Mat observationColor = Mat(1,256, CV_32F);
		read_contents(fileHandlerColor, observationColor, ausence);
		if(!ausence) {
			if(COMPARE_LOGIN)
				cout << compare(observationColor,logedUser) << endl;
			else {
				if(history.size() < LAST_X) {
					history.push_back(observationColor);
				}
				else {
					double mean = 0;
					for(int i = 0; i < history.size(); i++) {
						Mat x = history[i];
						mean += compare(observationColor,x);
					}
					history.pop_front();
					history.push_back(observationColor);
					cout << mean/LAST_X << endl;
				}
			}
		}
	}
	cout << "-----------------" << endl;
	for(int c = 1; c < args.size(); c+=1) {
		ifstream fileLoginColor(args[c]);
		fileLoginColor >> fileHandlerColor;
		//	auth cont stuff
		while(fileLoginColor >> fileHandlerColor) {
			Mat observationColor = Mat(1,256, CV_32F);
			read_contents(fileHandlerColor, observationColor, ausence);
			if(!ausence) {
				if(COMPARE_LOGIN)
					cout << compare(observationColor,logedUser) << endl;
				else {
					if(history.size() < LAST_X) {
						history.push_back(observationColor);
					}
					else {
						double mean = 0;
						for(int i = 0; i < history.size(); i++) {
							Mat x = history[i];
							mean += compare(observationColor,x);
						}
						history.pop_front();
						history.push_back(observationColor);
						cout << mean/LAST_X << endl;
					}
				}
			}
		}
		cout << "-----------------" << endl;
	}
	return 0;
}
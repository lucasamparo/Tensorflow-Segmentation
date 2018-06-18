#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace pcl;

double norm(float val, float max, float min, float start, float end){
	return (end-start)*((val - min)/(max - min))+start;
}

PointCloud<PointXYZRGBA>::Ptr deprojection(Mat depth, PointCloud<PointXYZRGBA>::Ptr model){
	PointCloud<PointXYZRGBA>::Ptr ret (new PointCloud<PointXYZRGBA>);
	int width = 128, height = 128;

	float maxX = -100000, minX = 100000, maxY = -100000, minY = 100000, maxZ = -10000, minZ = 10000;
	for(int i = 0; i < model->size(); i++){
		PointXYZRGBA p = model->points[i];
		if(p.z > maxZ)
			maxZ = p.z;
		if(p.z < minZ)
			minZ = p.z;
		if(p.x > maxX)
			maxX = p.x;
		if(p.x < minX)
			minX = p.x;
		if(p.y > maxY)
			maxY = p.y;
		if(p.y < minY)
			minY = p.y;
	}

	for(int i = 0; i < depth.size().width; i++){
		for(int j = 0; j < depth.size().height; j++){
			int d = depth.at<uchar>(j,i);
			if(d > 25){
				PointXYZRGBA p1;
				p1.x = ((i - 10)*(minX-maxX))/(width) + maxX;
				p1.y = ((j - 10)*(minY-maxY))/(height) + maxY;
				float f = (d - 127)/3.0;
				p1.z = f;
				p1.r = 255;
				p1.g = 255;
				p1.b = 255;
				ret->push_back(p1);
			}
		}
	}

	/*for(int i = 0; i < model->size(); i++){
		PointXYZRGBA p = model->points[i];
		int x = norm(p.x,minX,maxX,10,width-10);
		int y = norm(p.y,minY,maxY,10,height-10);
		int d = (int) depth.at<uchar>(y,x);
		PointXYZRGBA p1;
		p1.x = p.x;
		p1.y = p.y;
		float f = (d - 127)/3.0;
		p1.z = f;
		p1.r = 255;
		p1.g = 255;
		p1.b = 255;
		ret->push_back(p1);
	}*/

	return ret;
}

int main(int argc, char * argv[]){
	Mat depth = imread(argv[1],0);

	PointCloud<PointXYZRGBA>::Ptr model (new PointCloud<PointXYZRGBA>);
	if (io::loadPCDFile<PointXYZRGBA> ((argv[2]), *model) == -1){
		PCL_ERROR ("Couldn't read file a file\n");
	}

	PointCloud<PointXYZRGBA>::Ptr output (new PointCloud<PointXYZRGBA>);
	output = deprojection(depth,model);

	io::savePCDFile("deprojected.pcd",*output);

	return 0;
}

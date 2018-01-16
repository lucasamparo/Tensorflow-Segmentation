#include <detector.hpp>

#include <queue>

#include <util.hpp>

#define THRESHOLD_1 0.9
#define THRESHOLD_2 0.9
#define THRESHOLD_3 0.7
#define FACTOR 0.709
#define MINSIZE 90

#define FACE_HALF_SIZE 105
#define X_WIDTH 900.0
#define Y_WIDTH 800.0
#define RESOLUTION 0.127272727
#define DEPTH_FACE_SIZE 21
#define DEPTH_FACE_HALF_SIZE 10
#define PATH_CASCADE_FACE "reqs/ALL_Spring2003_3D.xml"

Detector::Detector(const std::string & model_path, const bool fast_resize) :
PNet(model_path + "/det1.prototxt", caffe::TEST),
RNet(model_path + "/det2.prototxt", caffe::TEST),
ONet(model_path + "/det3.prototxt", caffe::TEST),
thresholds({THRESHOLD_1, THRESHOLD_2, THRESHOLD_3}),
factor(FACTOR),
minsize(MINSIZE),
fastresize(fast_resize),
face_half_size(FACE_HALF_SIZE, FACE_HALF_SIZE, 0),
proj_height((int)(Y_WIDTH * RESOLUTION)),
proj_width((int)(X_WIDTH * RESOLUTION)),
proj_center(proj_width / 2, proj_height / 2),
min_x(0), max_x(30), min_y(-20), max_y(20) {
    // start caffe and its resources
    PNet.CopyTrainedLayersFrom(model_path + "/det1.caffemodel");
    RNet.CopyTrainedLayersFrom(model_path + "/det2.caffemodel");
    ONet.CopyTrainedLayersFrom(model_path + "/det3.caffemodel");
    
    // load depth detection resources
    // face_cascade = (CvHaarClassifierCascade *) cvLoad(PATH_CASCADE_FACE, 0, 0, 0);
    // if (!face_cascade)
    //     throw std::runtime_error("Error: unable to find face cascade used in depth detection");

    // p = cvCreateImage(cvSize(proj_width, proj_height), IPL_DEPTH_64F, 1);
    // m = cvCreateImage(cvSize(proj_width, proj_height), IPL_DEPTH_8U, 1);
    // sum = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_64F, 1);
    // sqsum = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_64F, 1);
    // tiltedsum = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_64F, 1);
    // sumint = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_32S, 1);
    // tiltedsumint = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_32S, 1);
    // msum = cvCreateImage(cvSize(proj_width + 1, proj_height + 1), IPL_DEPTH_32S, 1);
}

Detector::~Detector() {
	// cvReleaseImage(&p);
 //    cvReleaseImage(&m);
 //    cvReleaseImage(&sum);
 //    cvReleaseImage(&sqsum);
 //    cvReleaseImage(&tiltedsum);
 //    cvReleaseImage(&sumint);
 //    cvReleaseImage(&tiltedsumint);
 //    cvReleaseImage(&msum);
 //    cvReleaseHaarClassifierCascade(&face_cascade);
}

void Detector::generate_bounding_box(Eigen::MatrixXd & map, std::vector<Eigen::MatrixXd> & reg, double scale, double threshold, Eigen::MatrixXd & boxes) {
	assert(reg.size() == 4);

	int stride = 2;
    int cellsize = 12;
	
	Eigen::MatrixXd threshold_matrix = Eigen::MatrixXd(map.rows(), map.cols());
	threshold_matrix.fill(threshold);
	map -= threshold_matrix;
	map = map.cwiseMax(Eigen::MatrixXd::Zero(map.rows(), map.cols()));
	Eigen::MatrixXd I, J, V;
	igl::find(map, I, J, V); // I,J is index, V is value. They are all vectors

	// score
	threshold_matrix.resize(V.size(), 1);
	threshold_matrix.fill(threshold);
	Eigen::MatrixXd score = V + threshold_matrix;

	// reg
	Eigen::MatrixXd new_reg;
	new_reg.resize(I.size(), 4);
	for (int i = 0; i < 4; i++){ 
		Eigen::MatrixXd content = Eigen::MatrixXd::Zero(I.size(), 1);
		for (int num = 0; num < I.size(); num++){
			content(num) = reg[i](I(num), J(num));
		}
		new_reg.middleCols(i,1) = content;
	}
	
	// boundingbox
	Eigen::MatrixXd boundingbox;
	boundingbox.resize(I.size(), 2);
	boundingbox << I, J;

	Eigen::MatrixXd cellsize_m = Eigen::MatrixXd::Zero(boundingbox.rows(), boundingbox.cols());
	cellsize_m.fill(cellsize);

	Eigen::MatrixXd bb1 = (stride * boundingbox + Eigen::MatrixXd::Ones(boundingbox.rows(), boundingbox.cols())) / scale;
	Eigen::MatrixXd bb2 = (stride * boundingbox + cellsize_m) / scale;

	_fix(bb1);
	_fix(bb2);

	assert(bb1.rows() == bb2.rows());
	assert(bb1.rows() == score.rows());
	assert(bb1.rows() == new_reg.rows());
	assert(bb1.cols() == 2);
	assert(bb2.cols() == 2);
	assert(score.cols() == 1);
	assert(new_reg.cols() == 4);

	boxes.resize(bb1.rows(), 9);
	boxes << bb1, bb2, score, new_reg;
}

const std::vector<cv::Rect> & Detector::get_detections() const {
    return detections;
}

const cv::Rect & Detector::get_detection(const unsigned int idx) const {
    return detections[idx];
}

const std::vector<cv::Point> & Detector::get_landmarks(const unsigned int idx) const {
    return landmarks[idx];
}

const std::vector<std::vector<cv::Point>> & Detector::get_all_landmarks() const {
    return landmarks;
}

unsigned int Detector::num_faces() const {
    return detections.size();
}

void Detector::prepare_data1(const cv::Mat & img) {
    // 1. reshape data layer
	int height = img.rows;
	int width = img.cols;
	caffe::Blob<float> & input_layer = *PNet.input_blobs()[0];
	input_layer.Reshape(1, 3, height, width);

    // 2. link input data
	std::vector<cv::Mat> input_channels;
	float* input_data = input_layer.mutable_cpu_data();
	for (int i = 0, len = input_layer.channels(); i < len; ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}

	// 3. put img to data layer 
	cv::Mat sample_float;
	img.convertTo(sample_float, CV_32FC3);
	split(sample_float, input_channels);
}

void Detector::prepare_data2(const caffe::Net<float> & net, const std::vector<cv::Mat> & imgs) {
	assert(imgs.size() > 0);
	// 1. reshape data layer
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int numbox = imgs.size();
	caffe::Blob<float> & input_layer = *net.input_blobs()[0];
	input_layer.Reshape(numbox, 3, height, width);

	// 2. link input data and put into img data
	float * input_data = input_layer.mutable_cpu_data();
	const int input_layer_n_channels = input_layer.channels();
	for (int i = 0; i < numbox; i++){
		std::vector<cv::Mat> input_channels;
		for (int j = 0; j < input_layer_n_channels; j++){
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		split(imgs[i], input_channels);
	}
}

Eigen::MatrixXd Detector::stage1(cv::Mat & img_mat) {
    Eigen::MatrixXd total_boxes;
    total_boxes.resize(0, 9);
	int factor_count = 0;
	int h = img_mat.rows;
	int w = img_mat.cols;
	int minl = std::min(h, w);
	
	float m = 12.0 / minsize;
	minl *= m;

	// create scale pyramid
	std::vector<float> scales;
	while (minl >= 12) {
		scales.push_back(m * std::pow(factor, factor_count));
		minl *= factor;
		factor_count++;
	}

	for (auto scale : scales) {
		int hs = (int)std::ceil(h*scale);
		int ws = (int)std::ceil(w*scale);
		cv::Mat im_data;
		img_mat.convertTo(im_data, CV_32FC3);
		if (fastresize) { 
			im_data = (im_data - 127.5) * 0.0078125;
			resize(im_data, im_data, cv::Size(ws, hs));
		}
		else {
			resize(im_data, im_data, cv::Size(ws, hs));
			im_data = (im_data - 127.5) * 0.0078125;
		}

        cv::Mat im_t = cv::Mat(im_data.cols, im_data.rows, CV_32F);
        transpose(im_data, im_t);
        prepare_data1(im_t);

        PNet.Forward();

		caffe::Blob<float>* conv4_2 = PNet.output_blobs()[0]; // 1*4*height*width
		caffe::Blob<float>* prob1 = PNet.output_blobs()[1]; // 1*2*height*width

		Eigen::MatrixXd map; 
		std::vector<Eigen::MatrixXd> reg;
		convertToMatrix(prob1, conv4_2, map, reg);
		Eigen::MatrixXd boxes;
		generate_bounding_box(map, reg, scale, thresholds[0], boxes);

		if (boxes.rows() > 0) {
            std::vector<int> pick;
            nms(boxes, 0.5, "Union", pick);
            if (pick.size() > 0)
                _select(boxes, boxes, pick);
        }

        Eigen::MatrixXd t(total_boxes.rows() + boxes.rows(), boxes.cols());
		t << total_boxes, boxes;
		total_boxes.resize(t.rows(), t.cols());
		total_boxes << t;
	}
	return total_boxes;
}

void Detector::stage2(cv::Mat & img_mat, Eigen::MatrixXd & total_boxes) {
    cv::Mat im_data;
    img_mat.convertTo(im_data, CV_32FC3);

	std::vector<int> pick;
	nms(total_boxes, 0.7, "Union", pick);
	_select(total_boxes, total_boxes, pick);

	// using regression, convert n*9 to n*5
	Eigen::MatrixXd regh = total_boxes.middleCols(3, 1) - total_boxes.middleCols(1, 1);
	Eigen::MatrixXd regw = total_boxes.middleCols(2, 1) - total_boxes.middleCols(0, 1);
	Eigen::MatrixXd t1 = total_boxes.middleCols(0, 1) + regw.cwiseProduct(total_boxes.middleCols(5, 1));
	Eigen::MatrixXd t2 = total_boxes.middleCols(1, 1) + regh.cwiseProduct(total_boxes.middleCols(6, 1));
	Eigen::MatrixXd t3 = total_boxes.middleCols(2, 1) + regw.cwiseProduct(total_boxes.middleCols(7, 1));
	Eigen::MatrixXd t4 = total_boxes.middleCols(3, 1) + regh.cwiseProduct(total_boxes.middleCols(8, 1));
	Eigen::MatrixXd t5 = total_boxes.middleCols(4, 1);
	total_boxes.resize(total_boxes.rows(), 5);
    total_boxes << t1, t2, t3, t4, t5;
	rerec(total_boxes);

    Eigen::MatrixXd pad_params;
	pad(total_boxes, img_mat.cols, img_mat.rows, pad_params);
	// pad_params: 0 dy, 1 edy, 2 dx, 3 edx, 4 y, 5 ey, 6 x, 7 ex, 8 tmpw, 9 tmph;

	std::vector<cv::Mat> imgs;
	for (int i = 0; i < total_boxes.rows(); i++){
		cv::Mat tmp = cv::Mat::zeros(pad_params.col(9)[i], pad_params.col(8)[i], CV_32FC3);
        tmp = im_data(cv::Range(pad_params.col(4)[i], pad_params.col(5)[i] + 1),
			cv::Range(pad_params.col(6)[i], pad_params.col(7)[i] + 1));
		cv::Mat tmp_resize;
		resize(tmp, tmp_resize, cv::Size(24, 24));
		cv::Mat tmp_float;
		tmp_resize.convertTo(tmp_float, CV_32FC3);
		tmp_float = (tmp_float - 127.5) * 0.0078125;
        transpose(tmp_float, tmp_float);
		imgs.push_back(tmp_float);
    }

    prepare_data2(RNet, imgs);

    RNet.Forward();
    caffe::Blob<float> * conv5_2 = RNet.output_blobs()[0];
	caffe::Blob<float> * prob1 = RNet.output_blobs()[1]; 

	//use prob1 to filter total_boxes 
    std::vector<double> score;
    convertToVector(prob1, score);

    std::vector<int> pass_t;
	_find(score, thresholds[1], pass_t);

    filter(total_boxes, pass_t, score);

    // use conv5-2 to bbreg
	Eigen::MatrixXd mv;
	getMV(conv5_2, mv, pass_t);  // 4*N
    if (total_boxes.rows() > 0){
        bbreg(total_boxes, mv);
        std::vector<int> pick;
        nms(total_boxes, 0.5, "Union", pick);
        if (pick.size() > 0){
			_select(total_boxes, total_boxes, pick);
        }

		rerec(total_boxes);
	}
}

void Detector::stage3(cv::Mat & img_mat, Eigen::MatrixXd & total_boxes) {
	Eigen::MatrixXd pad_params;
	pad(total_boxes, img_mat.cols, img_mat.rows, pad_params);
	// pad_params: 0 dy, 1 edy, 2 dx, 3 edx, 4 y, 5 ey, 6 x, 7 ex, 8 tmpw, 9 tmph;
	
	std::vector<cv::Mat> imgs;
	for (int i = 0; i < total_boxes.rows(); i++){
		cv::Mat tmp = cv::Mat::zeros(pad_params.col(9)[i], pad_params.col(8)[i], CV_32FC3);
		tmp = img_mat(cv::Range(pad_params.col(4)[i], pad_params.col(5)[i] + 1), cv::Range(pad_params.col(6)[i], pad_params.col(7)[i] + 1));
		cv::Mat tmp_resize;
		cv::resize(tmp, tmp_resize, cv::Size(48, 48));
		cv::Mat tmp_float;
		tmp_resize.convertTo(tmp_float, CV_32FC3);
		tmp_float = (tmp_float - 127.5) * 0.0078125;
		imgs.push_back(tmp_float.t());
    }

	prepare_data2(ONet, imgs);
    ONet.Forward();
    caffe::Blob<float> * conv6_2 = ONet.output_blobs()[0]; // 4
    caffe::Blob<float> * conv6_3 = ONet.output_blobs()[1]; // 10
    caffe::Blob<float> * prob1 = ONet.output_blobs()[2]; // 2
    
    cv::Mat points(conv6_3->shape(0), conv6_3->shape(1), CV_32F, conv6_3->mutable_cpu_data());
    
	// use prob1 to filter total_boxes 
	std::vector<double> score;

    convertToVector(prob1, score);
	std::vector<int> pass_t;
	_find(score, thresholds[2], pass_t);
    for (int i : pass_t)
	    for (int j = 0; j < points.cols / 2; j++) {
    	    points.at<float>(i, j) *= (total_boxes(i, 2) - total_boxes(i, 0));
    	    points.at<float>(i, j) += total_boxes(i, 0);
    	    points.at<float>(i, j + points.cols / 2) *= (total_boxes(i, 3) - total_boxes(i, 1));
    	    points.at<float>(i, j + points.cols / 2) += total_boxes(i, 1);
	    }
    filter(total_boxes, pass_t, score);
	
	// use conv6-2 to bbreg
	Eigen::MatrixXd mv;
	getMV(conv6_2, mv, pass_t);  
	if (total_boxes.rows() > 0){ 
        bbreg(total_boxes, mv);

		std::vector<int> pick;
        nms(total_boxes, 0.5, "Min", pick);
        if (pick.size() > 0) {
			_select(total_boxes, total_boxes, pick);
			for (int i : pick) {
			    std::vector<cv::Point> landmark;
			    for (int j = 0; j < points.cols / 2; j++)
    			    landmark.emplace_back(points.at<float>(pass_t[i], j), points.at<float>(pass_t[i], j + points.cols / 2));
			    landmarks.push_back(landmark);
		    }
	    }
	}
}

std::vector<cv::Point3d> Detector::grid_sampling(const cv::Rect & det, const Camera & cam, const double thr, const int jump) {
	std::vector<cv::Point3d> ret;
	for (int i = det.y; i < det.y + det.height; i += jump)
		for (int j = det.x; j < det.x + det.width; j += jump) {
			const cv::Point3d p = cam.depthToXyz(i, j);
			if (p.z < -1.0 && p.z > -thr)
				ret.push_back(p * RESOLUTION);
		}
	return ret;
}

cv::Point3d Detector::depth_landmark(const cv::Point & landmark, const Camera & cam) {
	cv::Point depth_nose(0, 0);
	cv::Point3d face_coords(0, 0, 0);
	cv::Point corr_nose(0, 0);
	for (int i = 0; i < cam.depth.rows; i++)
		for (int j = 0; j < cam.depth.cols; j++) {
			const cv::Point3d p = cam.depthToXyz(i, j);
			const cv::Point cand = cam.xyzToColor(p);
			if ((landmark.x - cand.x)*(landmark.x - cand.x) + (landmark.y - cand.y)*(landmark.y - cand.y) < (landmark.x - corr_nose.x)*(landmark.x - corr_nose.x) + (landmark.y - corr_nose.y)*(landmark.y - corr_nose.y)) {
				corr_nose = cand;
				face_coords = p;
				depth_nose = cv::Point(j, i);
			}
		}
	if (depth_nose.x > 0 && depth_nose.y > 0 && depth_nose.x < cam.depth.cols && depth_nose.y < cam.depth.rows)
		return face_coords;
	return cv::Point3d(0, 0, 0);
}

void Detector::compute_rotation_matrix(double matrix[3][3], double imatrix[3][3], const double aX, const double aY, const double aZ) {
    double cosX, cosY, cosZ, sinX, sinY, sinZ;

    cosX = cos(aX);
    cosY = cos(aY);
    cosZ = cos(aZ);
    sinX = sin(aX);
    sinY = sin(aY);
    sinZ = sin(aZ);

    matrix[0][0] = cosZ*cosY+sinZ*sinX*sinY;
    matrix[0][1] = sinZ*cosY-cosZ*sinX*sinY;
    matrix[0][2] = cosX*sinY;
    matrix[1][0] = -sinZ*cosX;
    matrix[1][1] = cosZ*cosX;
    matrix[1][2] = sinX;
    matrix[2][0] = sinZ*sinX*cosY-cosZ*sinY;
    matrix[2][1] = -cosZ*sinX*cosY-sinZ*sinY;
    matrix[2][2] = cosX*cosY;

    for (int i = 0; i < 3; i++)
    	for (int j = 0; j < 3; j++)
    		imatrix[i][j] = matrix[j][i];
}

void Detector::compute_projection(const std::vector<cv::Point3d> & points, double matrix[3][3], const double background) {
    // compute projection
    cvSet(p, cvRealScalar(-DBL_MAX));
    cvSet(m, cvRealScalar(0));
    for (const cv::Point3d & point : points) {
        const int i = proj_center.y - cvRound(point.x * matrix[1][0] + point.y * matrix[1][1] + point.z * matrix[1][2]);
        const int j = proj_center.x + cvRound(point.x * matrix[0][0] + point.y * matrix[0][1] + point.z * matrix[0][2]);
        const double d = point.x * matrix[2][0] + point.y * matrix[2][1] + point.z * matrix[2][2];

        if (i >= 0 && j >= 0 && i < proj_height && j < proj_width && d > CV_IMAGE_ELEM(p, double, i, j)) {
            CV_IMAGE_ELEM(p, double, i, j) = d;
            CV_IMAGE_ELEM(m, uchar, i, j) = 1;
        }
    }
    
    // hole filling
    std::queue<int> li, lj, lc;
    for (int i = 1; i < proj_height - 1; i++)
        for (int j = 1; j < proj_width - 1; j++)
            if (!CV_IMAGE_ELEM(m, uchar, i, j) && (CV_IMAGE_ELEM(m, uchar, i, j - 1) || CV_IMAGE_ELEM(m, uchar, i, j + 1) || CV_IMAGE_ELEM(m, uchar, i - 1, j) || CV_IMAGE_ELEM(m, uchar, i + 1, j))) {
                li.push(i);
                lj.push(j);
                lc.push(1);
            }

    while (!li.empty()) {
        int i = li.front(), j = lj.front(), c = lc.front();
        li.pop(), lj.pop(), lc.pop();
        
        if (!CV_IMAGE_ELEM(m, uchar, i, j) && i > 0 && i < proj_height - 1 && j > 0 && j < proj_width - 1 && c < DEPTH_FACE_HALF_SIZE) {
            CV_IMAGE_ELEM(m, uchar, i, j) = c + 1;
            int t = 0;
            double d = 0;
            if (CV_IMAGE_ELEM(m, uchar, i, j - 1) && CV_IMAGE_ELEM(m, uchar, i, j - 1) <= c) {
                t++;
                d += CV_IMAGE_ELEM(p, double, i, j - 1);
            }
            else {
                li.push(i);
                lj.push(j - 1);
                lc.push(c + 1);
            }
            if (CV_IMAGE_ELEM(m, uchar, i, j + 1) && CV_IMAGE_ELEM(m, uchar, i, j + 1) <= c) {
                t++;
                d += CV_IMAGE_ELEM(p, double, i, j + 1);
            }
            else {
                li.push(i);
                lj.push(j + 1);
                lc.push(c + 1);
            }
            if (CV_IMAGE_ELEM(m, uchar, i - 1, j) && CV_IMAGE_ELEM(m, uchar, i - 1, j) <= c) {
                t++;
                d += CV_IMAGE_ELEM(p, double, i - 1, j);
            }
            else {
                li.push(i - 1);
                lj.push(j);
                lc.push(c + 1);
            }
            if (CV_IMAGE_ELEM(m, uchar, i + 1, j) && CV_IMAGE_ELEM(m, uchar, i + 1, j) <= c) {
                t++;
                d += CV_IMAGE_ELEM(p, double, i + 1, j);
            }
            else {
                li.push(i + 1);
                lj.push(j);
                lc.push(c + 1);
            }
            CV_IMAGE_ELEM(p, double, i, j) = d / (double)t;
        }
    }

    // final adjustments
    for (int i = 0; i < proj_height; i++)
        for (int j = 0; j < proj_width; j++) {
            if (CV_IMAGE_ELEM(p, double, i, j) == -DBL_MAX)
                CV_IMAGE_ELEM(p, double, i, j) = background;
            if (CV_IMAGE_ELEM(m, uchar, i, j))
                CV_IMAGE_ELEM(m, uchar, i, j) = 1;
        }
}

bool Detector::depth_verification(const std::vector<cv::Point3d> & points) {
	double matrix[3][3], imatrix[3][3];

	// detect in multiple poses
	for (int ax = min_x; ax <= max_x; ax += 10) {
		for (int ay = min_y; ay <= max_y; ay += 10) {
		    compute_rotation_matrix(matrix, imatrix, ax * 0.017453293, ay * 0.017453293, 0);
		    compute_projection(points, matrix);
		    
		    cvIntegral(p, sum, sqsum, tiltedsum);
		    cvIntegral(m, msum, NULL, NULL);

		    for (int i = 0; i <= proj_height; i++)
		        for (int j = 0; j <= proj_width; j++) {
		            CV_IMAGE_ELEM(sumint, int, i, j) = CV_IMAGE_ELEM(sum, double, i, j);
		            CV_IMAGE_ELEM(tiltedsumint, int, i, j) = CV_IMAGE_ELEM(tiltedsum, double, i, j);
		        }
		    
		    cvSetImagesForHaarClassifierCascade(face_cascade, sumint, sqsum, tiltedsumint, 1.0);
		    
		    for (int i = 0; i <= proj_height - DEPTH_FACE_SIZE; i++)
		        for (int j = 0; j <= proj_width - DEPTH_FACE_SIZE; j++)
		            if (CV_IMAGE_ELEM(msum, int, i + DEPTH_FACE_SIZE, j + DEPTH_FACE_SIZE) - CV_IMAGE_ELEM(msum, int, i, j + DEPTH_FACE_SIZE) - CV_IMAGE_ELEM(msum, int, i + DEPTH_FACE_SIZE, j) + CV_IMAGE_ELEM(msum, int, i, j) == 441 && cvRunHaarClassifierCascade(face_cascade, cvPoint(j, i), 0) > 0)
	            		return true;
		}
	}		
	
	return false;
}

const cv::Point3d & Detector::get_depth_detection() const {
    return best_face_coords;
}

bool Detector::is_face(const unsigned int idx, const Camera & cam) {
	if (num_faces() <= idx)
		throw std::runtime_error("Error: out of bounds detection indice");

	// detect nose in depth
	best_face_coords = depth_landmark(landmarks[idx][2], cam);
	if (best_face_coords.z == 0)
		return false;
	
	// crop image according to depth
	cv::Rect depth_detection(cam.xyzToDepth(best_face_coords + face_half_size), cam.xyzToDepth(best_face_coords - face_half_size));
	
	// grid sample to detect
	std::vector<cv::Point3d> points = grid_sampling(depth_detection, cam);
	
	// detect in depth crop
	return depth_verification(points);
}

void Detector::operator()(const cv::Mat & _img) {
    landmarks.clear();
    detections.clear();
    
    // convert img to RGB
    cv::Mat img;
    _img.copyTo(img);
    cvtColor(img, img, CV_BGR2RGB);

    // detect faces
    Eigen::MatrixXd total_boxes = stage1(img);
	
    if (total_boxes.rows() > 0) {
        stage2(img, total_boxes);
        
        if (total_boxes.rows() > 0)
            stage3(img, total_boxes);
    }
    
    // turn detection into rects
    for (int i = 0, len = total_boxes.rows(); i < len; i++)
        detections.emplace_back(cv::Point(total_boxes(i, 0), total_boxes(i, 1)), cv::Point(total_boxes(i, 2), total_boxes(i, 3)));
}

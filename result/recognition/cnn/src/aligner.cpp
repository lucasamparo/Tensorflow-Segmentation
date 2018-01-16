#include <aligner.hpp>

double Aligner::walk(double X, double Y, double Z, int i, int j, int c, double dist) {
	if (c < 30) {
		double d, dmin;
		int it = 0, jt = 0;

		dmin = dist;
		if (i > 0) {
			d = pow(X - avg.at<cv::Vec3d>(i - 1, j)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i - 1, j)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i - 1, j)[2], 2.0);
			if (d < dmin) {
				dmin = d;
				it = -1;
				jt = 0;
			}
			if (j > 0) {
				d = pow(X - avg.at<cv::Vec3d>(i - 1, j - 1)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i - 1, j - 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i - 1, j - 1)[2], 2.0);
				if (d < dmin) {
					dmin = d;
					it = -1;
					jt = -1;
				}
			}
			if (j < avg.cols - 1) {
				d = pow(X - avg.at<cv::Vec3d>(i - 1, j + 1)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i - 1, j + 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i - 1, j + 1)[2], 2.0);
				if (d < dmin) {
					dmin = d;
					it = -1;
					jt = 1;
				}
			}
		}
		if (i < avg.rows - 1) {
			d = pow(X - avg.at<cv::Vec3d>(i + 1, j)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i + 1, j)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i + 1, j)[2], 2.0);
			if (d < dmin) {
				dmin = d;
				it = 1;
				jt = 0;
			}
			if (j > 0) {
				d = pow(X - avg.at<cv::Vec3d>(i + 1, j - 1)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i + 1, j - 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i + 1, j - 1)[2], 2.0);
				if (d < dmin) {
					dmin = d;
					it = 1;
					jt = -1;
				}
			}
			if (j < avg.cols - 1) {
				d = pow(X - avg.at<cv::Vec3d>(i + 1, j + 1)[0], 2.0)+pow(Y - avg.at<cv::Vec3d>(i + 1, j + 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i + 1, j + 1)[2], 2.0);
				if (d < dmin) {
					dmin = d;
					it = 1;
					jt = 1;
				}
			}
		}
		if (j > 0) {
			d = pow(X - avg.at<cv::Vec3d>(i, j - 1)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i, j - 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i, j - 1)[2], 2.0);
			if (d < dmin) {
				dmin = d;
				it = 0;
				jt = -1;
			}
		}
		if (j < avg.cols - 1) {
			d = pow(X - avg.at<cv::Vec3d>(i, j + 1)[0], 2.0) + pow(Y - avg.at<cv::Vec3d>(i, j + 1)[1], 2.0) + pow(Z - avg.at<cv::Vec3d>(i, j + 1)[2], 2.0);
			if (d < dmin) {
				dmin = d;
				it = 0;
				jt = 1;
			}
		}

		if (dmin < dist)
			return walk(X, Y, Z, i + it, j + jt, c + 1, dmin);
		else {
			row = i;
			col = j;
			return dist;
		}
	}
	else {
		row = i;
		col = j;
		return dist;
	}
}

void Aligner::load_avg_face_model() {
    int dim[4], size = height * width;
	std::ifstream fp(AVERAGE_MODEL_PATH);
	if (!fp)
		throw std::runtime_error("Error: average face model is missing");

	fp.read((char *) dim, 4*sizeof(int));
	if (dim[0] != height || dim[1] != width || dim[2] != proj_origin.y || dim[3] != proj_origin.x)
	    throw std::runtime_error("Error: average face model has invalid dimensions!");

	avg.create(height, width, CV_64FC3);
	fp.read((char *) avg.data, size*3*sizeof(double));
}

Aligner::Aligner() :
    proj_origin(cvRound(MODEL_WIDTH/MODEL_RESOLUTION), cvRound(MODEL_HEIGHT_1/MODEL_RESOLUTION)),
    final_proj_origin(cvRound(320/MODEL_RESOLUTION), cvRound(240/MODEL_RESOLUTION)),
    width(2*cvRound(MODEL_WIDTH/MODEL_RESOLUTION) + 1),
	height(cvRound(MODEL_HEIGHT_1/MODEL_RESOLUTION) + cvRound(MODEL_HEIGHT_2/MODEL_RESOLUTION) + 1) {
	
	load_avg_face_model();
}

Aligner::Aligner(const std::vector<cv::Point3d> & cloud, const cv::Point3d & face_center) :
    proj_origin(cvRound(MODEL_WIDTH/MODEL_RESOLUTION), cvRound(MODEL_HEIGHT_1/MODEL_RESOLUTION)),
    final_proj_origin(cvRound(320/MODEL_RESOLUTION), cvRound(240/MODEL_RESOLUTION)),
    width(2*cvRound(MODEL_WIDTH/MODEL_RESOLUTION) + 1),
	height(cvRound(MODEL_HEIGHT_1/MODEL_RESOLUTION) + cvRound(MODEL_HEIGHT_2/MODEL_RESOLUTION) + 1) {
	
	load_avg_face_model();
	align(cloud, face_center);
}

Aligner::Aligner(const Camera & calibration, const cv::Vec6d & argtransf, const std::vector<cv::Point>  & landmarks2d) : Aligner() {
    align_registered(calibration, argtransf, landmarks2d);
}

void Aligner::align(const std::vector<cv::Point3d> & cloud, const cv::Point3d & face_center) {
	if (find_alignment(cloud, face_center))
	    compute_depth_map(cloud);
    else
        throw std::runtime_error("Error: unable to align");
}

void Aligner::align_registered(const Camera & calibration, const cv::Vec6d & argtransf, const std::vector<cv::Point> & landmarks2d) {
    // compute alignment tranformation
	double matrix[3][3];
	cv::Point3d tmp_point, translation_vector(argtransf[0], argtransf[1], argtransf[2]);
	compute_rotation_matrix(matrix, argtransf[3], argtransf[4], argtransf[5]);
	
	// create images
	img.create(calibration.depth.size(), CV_8U);
	img.setTo(0);
	cv::Mat final_dcorr(calibration.depth.size(), CV_64FC3);
	final_dcorr.setTo(DBL_MAX);
	cv::Mat final_corr(calibration.depth.size(), CV_32S);
	final_corr.setTo(0);
	
	// find proper points
	for (int i = 0; i < calibration.depth.rows; i++) {
	    for (int j = 0; j < calibration.depth.cols; j++) {
    	    // project depth point
    	    const cv::Point3d point = calibration.depthToXyz(i, j);
			if (point.z < -1.0 && point.z > -DEPTH_THRESHOLD) {
				// deproject point for proper alignment
                cv::Point3d tmp_point = transform_point(matrix, translation_vector, point);
		        row = cvRound(-tmp_point.y/MODEL_RESOLUTION + final_proj_origin.y);
		        col = cvRound(tmp_point.x/MODEL_RESOLUTION + final_proj_origin.x);
            
                // draw accordingly
                if (row >= 0 && col >= 0 && row < img.rows && col < img.cols) {
	                final_dcorr.at<cv::Vec3d>(row, col) = tmp_point;
			        final_corr.at<int>(row, col) = 1;
                }
			}
        }
    }

	hole_filling(final_corr, final_dcorr);
	
	// compute inverse rotation matrix (by translation)
	double imatrix[3][3];
	for (int i = 0; i < 3; i++)
	    for (int j = 0; j < 3; j++)
	        imatrix[i][j] = matrix[j][i];
    
    // resize resulting landmarks vector
    std::vector<cv::Point> projected_landmarks;
    const int n_landmarks = landmarks2d.size();
    landmarks.resize(n_landmarks);
    ir_landmarks.resize(n_landmarks);
    for (int i = 0; i < n_landmarks; i++)
        projected_landmarks.emplace_back(-1, -1);
    
	// find closest match to landmarks
	for (int i = 0; i < img.rows; i++)
	    for (int j = 0; j < img.cols; j++)
	        if (final_corr.at<int>(i, j)) {
	            // draw final projection
	            double z = 127.0 + 3.0*final_dcorr.at<cv::Vec3d>(i, j)[2];
	            img.at<uchar>(i, j) = (z < 0.0 ? 0 : (z > 255.0 ? 255 : z));
	            
	            // project hole-filled image
	            z = final_dcorr.at<cv::Vec3d>(i, j)[2];
	            cv::Point3d tmp_point = transform_point(imatrix, cv::Point3d(0, 0, 0), cv::Point3d(MODEL_RESOLUTION*j - final_proj_origin.x, -MODEL_RESOLUTION*i + final_proj_origin.y, z) - translation_vector);
			
			    // deproject into rgb img
			    const cv::Point rgb_point = calibration.xyzToColor(tmp_point);
			
			    // update closest match to landmarks
			    for (int k = 0; k < n_landmarks; k++)
			        if (projected_landmarks[k] == cv::Point(-1, -1) || pow(rgb_point.x - landmarks2d[k].x, 2.0) + pow(rgb_point.y - landmarks2d[k].y, 2.0) < pow(projected_landmarks[k].x - landmarks2d[k].x, 2.0) + pow(projected_landmarks[k].y - landmarks2d[k].y, 2.0)) {
			            landmarks[k] = cv::Point(j, i);
		                ir_landmarks[k] = calibration.xyzToDepth(tmp_point);
			            projected_landmarks[k] = rgb_point;
		            }
	        }
}

void Aligner::align_registered(const std::vector<cv::Point3d> & points, const std::vector<cv::Point2d> & pixels, const cv::Point3d & face_center) {
	if (find_alignment(points, face_center))
	    compute_registered_depth_map(points, pixels);
    else
        throw std::runtime_error("Error: unable to align");
}

bool Aligner::find_alignment(const std::vector<cv::Point3d> & cloud, const cv::Point3d & face_center) {
    prep_ninja_mask(cloud, face_center);
    
	if (ninja_mask_cloud.empty())
	    return false;

	find_pre_alignment();
	iterative_closest_points();
	return true;
}

void Aligner::prep_ninja_mask(const std::vector<cv::Point3d> & cloud, const cv::Point3d & face_center) {
	ninja_mask_cloud.clear();
	const double x_min = face_center.x - 100*ROI_WIDTH, x_max = face_center.x + 100*ROI_WIDTH;
	const double y_min = face_center.y - 100*ROI_HEIGHT, y_max = face_center.y + 100*ROI_HEIGHT;
	for (const cv::Point3d & p : cloud)
		if (p.x >= x_min && p.x <= x_max && p.y >= y_min && p.y <= y_max)
            ninja_mask_cloud.emplace_back(p.x, p.y, p.z);
}

void Aligner::find_pre_alignment() {
    int n = ninja_mask_cloud.size();

    std::vector<double> buffer;
	for (const cv::Point3d & point : ninja_mask_cloud)
		buffer.push_back(point.x);
	n /= 2;
	std::nth_element(buffer.begin(), buffer.begin() + n, buffer.end());
	transf[0] = -buffer[n];

	buffer.clear();
	for (const cv::Point3d & point : ninja_mask_cloud)
		buffer.push_back(point.z);
	std::nth_element(buffer.begin(), buffer.begin() + n, buffer.end());
	transf[2] = -buffer[n];

	buffer.clear();
	for (const cv::Point3d & point : ninja_mask_cloud)
		buffer.push_back(point.y);
	n /= 2;
	std::nth_element(buffer.begin(), buffer.begin() + n, buffer.end());
	transf[1] = -buffer[n];

	transf[3] = transf[4] = transf[5] = 0.0;
}

void Aligner::compute_rotation_matrix(double matrix[3][3], double teta_x, double teta_y, double teta_z) {
	double sin_x = sin(teta_x), cos_x = cos(teta_x);
	double sin_y = sin(teta_y), cos_y = cos(teta_y);
	double sin_z = sin(teta_z), cos_z = cos(teta_z);

	matrix[0][0] = (cos_z * cos_y) + (sin_z * sin_x * sin_y);
	matrix[0][1] = (sin_z * cos_y) - (cos_z * sin_x * sin_y);
	matrix[0][2] = cos_x * sin_y;
	matrix[1][0] = -(sin_z * cos_x);
	matrix[1][1] = cos_z * cos_x;
	matrix[1][2] = sin_x;
	matrix[2][0] = (sin_z * sin_x * cos_y) - (cos_z * sin_y);
	matrix[2][1] = -(cos_z * sin_x * cos_y) - (sin_z * sin_y);
	matrix[2][2] = cos_x * cos_y;
}

cv::Point3d Aligner::transform_point(double rotation_matrix[3][3], const cv::Point3d & translation_vector, const cv::Point3d & old_point) {
    cv::Point3d new_point;
	new_point.x = (old_point.x * rotation_matrix[0][0]) + (old_point.y * rotation_matrix[0][1]) + (old_point.z * rotation_matrix[0][2]);
	new_point.y = (old_point.x * rotation_matrix[1][0]) + (old_point.y * rotation_matrix[1][1]) + (old_point.z * rotation_matrix[1][2]);
	new_point.z = (old_point.x * rotation_matrix[2][0]) + (old_point.y * rotation_matrix[2][1]) + (old_point.z * rotation_matrix[2][2]);
	new_point += translation_vector;
	return new_point;
}

double Aligner::project_and_walk(const cv::Point3d & p) {
	int i = cvRound(-p.y/MODEL_RESOLUTION + proj_origin.y);
	if (i < 0 || i >= avg.rows)
		return DBL_MAX;

	int j = cvRound(p.x/MODEL_RESOLUTION + proj_origin.x);
	if (j < 0 || j >= avg.cols)
		return DBL_MAX;

	return walk(p.x, p.y, p.z, i, j, 0, pow(p.x - avg.at<cv::Vec3d>(i,j)[0], 2.0) + pow(p.y - avg.at<cv::Vec3d>(i,j)[1], 2.0) + pow(p.z - avg.at<cv::Vec3d>(i,j)[2], 2.0));
}

double Aligner::iterative_closest_points(const int max_iterations, const double outl_sqrd_thr) {
    double rot_matrix[3][3];
	double err, old_err, best_err;
	cv::Point3d transfd_point;
	cv::Vec6d best_transf;
	cv::Mat dcorr(avg.size(), CV_64F), corr(avg.size(), CV_32S);
	std::vector<cv::Point3d> ref_pts, mov_pts;
	err = old_err = best_err = DBL_MAX;

	for (int count = 0, wcount = 0, ucount = 0;; count++) {
		cv::Point3d translation_vector(transf[0], transf[1], transf[2]);
		compute_rotation_matrix(rot_matrix, transf[3], transf[4], transf[5]);
		dcorr.setTo(DBL_MAX);
		for (int i = 0, len = ninja_mask_cloud.size(); i < len; i++) {
			transfd_point = transform_point(rot_matrix, translation_vector, ninja_mask_cloud[i]);
			double d = project_and_walk(transfd_point);
			if (d < outl_sqrd_thr && row > 0 && col > 0 && row < dcorr.rows - 1 && col < dcorr.cols - 1 && d < dcorr.at<double>(row, col)) {
				dcorr.at<double>(row, col) = d;
				corr.at<int>(row, col) = i;
			}
		}

		err = 0.0;
		for (int i = 0; i < dcorr.rows; i++)
			for (int j = 0; j < dcorr.cols; j++)
				if (dcorr.at<double>(i, j) != DBL_MAX) {
					ref_pts.push_back(avg.at<cv::Vec3d>(i, j));
					mov_pts.push_back(ninja_mask_cloud[corr.at<int>(i, j)]);
					err += dcorr.at<double>(i, j);
				}
		err /= ref_pts.size();

		if (err < best_err) {
			best_transf = transf;
			best_err = err;
			ucount = 0;
		}
		else {
			ucount++;
		    if (ucount > 1)
			    break;
	    }

		if (err == 0.0 || err >= old_err)
			break;

		if (err + 0.001 > old_err) {
			wcount++;
			if (wcount > 2)
    			break;
		}
		else
			wcount = 0;

		old_err = err;

		horn(transf, ref_pts, mov_pts);
		
		ref_pts.clear();
		mov_pts.clear();
	}
	
	transf = best_transf;

	return best_err;
}

void Aligner::invert_3x3_matrix(double m[3][3], double i[3][3]) const {
	double a11, a12, a13, a21, a22, a23, a31, a32, a33, det;

	a11 = m[0][0];
	a12 = m[0][1];
	a13 = m[0][2];
	a21 = m[1][0];
	a22 = m[1][1];
	a23 = m[1][2];
	a31 = m[2][0];
	a32 = m[2][1];
	a33 = m[2][2];

	det = a11*(a33*a22 - a32*a23) - a21*(a33*a12 - a32*a13) + a31*(a23*a12 - a22*a13);

	i[0][0] = (a33*a22 - a32*a23) / det;
	i[0][1] = -(a33*a12 - a32*a13) / det;
	i[0][2] = (a23*a12 - a22*a13) / det;
	i[1][0] = -(a33*a21 - a31*a23) / det;
	i[1][1] = (a33*a11 - a31*a13) / det;
	i[1][2] = -(a23*a11 - a21*a13) / det;
	i[2][0] = (a32*a21 - a31*a22) / det;
	i[2][1] = -(a32*a11 - a31*a12) / det;
	i[2][2] = (a22*a11 - a21*a12) / det;
}

void Aligner::sqrt_3x3_matrix(double A[3][3], double Y[3][3]) const {
	double YI[3][3], Z[3][3], ZI[3][3], D[3][3], error;
	int i, j, k, count;

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			Y[i][j] = A[i][j];
			if (i == j)
				Z[i][j] = 1.0;
			else
				Z[i][j] = 0.0;
		}
	}

	error = 0.0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			D[i][j] = 0.0;
			for (k = 0; k < 3; k++)
				D[i][j] += Y[i][k]*Y[k][j];
			error += pow(D[i][j]-A[i][j], 2.0);
		}
	}

	count=0;
	while (error > 10E-55 && count < 5000) {
		this->invert_3x3_matrix(Y, YI);
		this->invert_3x3_matrix(Z, ZI);

		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				Y[i][j] = (Y[i][j]+ZI[i][j])/2.0;
				Z[i][j] = (Z[i][j]+YI[i][j])/2.0;
			}
		}

		error = 0.0;
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				D[i][j] = 0.0;
				for (k = 0; k < 3; k++)
					D[i][j] += Y[i][k]*Y[k][j];
				error += fabs(D[i][j]-A[i][j]);
			}
		}

		count++;
	}
}

void Aligner::horn(cv::Vec6d & transf, const std::vector<cv::Point3d> & ref_pts, const std::vector<cv::Point3d> & mov_pts) {
	cv::Point3d mov_sum(0.0, 0.0, 0.0), ref_sum(0.0, 0.0, 0.0);
	double M[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, M1[3][3], M2[3][3], ax, ay, az, sx, sy, sz, cx, cy, cz, m0, m1, m2;

	const int len = ref_pts.size();
	for (int i = 0; i < len; i++) {
	    mov_sum += mov_pts[i];
	    ref_sum += ref_pts[i];

		M[0][0] += mov_pts[i].x * ref_pts[i].x;
		M[0][1] += mov_pts[i].y * ref_pts[i].x;
		M[0][2] += mov_pts[i].z * ref_pts[i].x;
		M[1][0] += mov_pts[i].x * ref_pts[i].y;
		M[1][1] += mov_pts[i].y * ref_pts[i].y;
		M[1][2] += mov_pts[i].z * ref_pts[i].y;
		M[2][0] += mov_pts[i].x * ref_pts[i].z;
		M[2][1] += mov_pts[i].y * ref_pts[i].z;
		M[2][2] += mov_pts[i].z * ref_pts[i].z;
	}

	M[0][0] -= (mov_sum.x * ref_sum.x) / (double) len;
	M[0][1] -= (mov_sum.y * ref_sum.x) / (double) len;
	M[0][2] -= (mov_sum.z * ref_sum.x) / (double) len;
	M[1][0] -= (mov_sum.x * ref_sum.y) / (double) len;
	M[1][1] -= (mov_sum.y * ref_sum.y) / (double) len;
	M[1][2] -= (mov_sum.z * ref_sum.y) / (double) len;
	M[2][0] -= (mov_sum.x * ref_sum.z) / (double) len;
	M[2][1] -= (mov_sum.y * ref_sum.z) / (double) len;
	M[2][2] -= (mov_sum.z * ref_sum.z) / (double) len;

	mov_sum.x /= (double) len;
	mov_sum.y /= (double) len;
	mov_sum.z /= (double) len;
	ref_sum.x /= (double) len;
	ref_sum.y /= (double) len;
	ref_sum.z /= (double) len;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			M1[i][j] = 0.0;
			for (int k = 0; k < 3; k++)
				M1[i][j] += M[k][i]*M[k][j];
		}
	}

	invert_3x3_matrix(M1, M2);
	sqrt_3x3_matrix(M2, M1);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			M2[i][j] = 0.0;
			for (int k = 0; k < 3; k++)
				M2[i][j] += M[i][k]*M1[k][j];
		}
	}

	transf[3] = ax = asin(M2[1][2]);
	transf[4] = ay = atan2(M2[0][2], M2[2][2]);
	transf[5] = az = atan2(-M2[1][0], M2[1][1]);

	sx = sin(ax); cx = cos(ax);
	sy = sin(ay); cy = cos(ay);
	sz = sin(az); cz = cos(az);

	m0 = (cz * cy) + (sz * sx * sy);
	m1 = (sz * cy) - (cz * sx * sy);
	m2 = cx * sy;
	transf[0] = ref_sum.x - ((mov_sum.x * m0) + (mov_sum.y * m1) + (mov_sum.z * m2));
	m0 = -sz * cx;
	m1 = cz * cx;
	m2 = sx;
	transf[1] = ref_sum.y - ((mov_sum.x * m0) + (mov_sum.y * m1) + (mov_sum.z * m2));
	m0 = (sz*sx*cy) - (cz * sy);
	m1 = -(cz * sx * cy) - (sz * sy);
	m2 = cx * cy;
	transf[2] = ref_sum.z - ((mov_sum.x * m0) + (mov_sum.y * m1) + (mov_sum.z * m2));
}

void Aligner::hole_filling(cv::Mat & final_corr, cv::Mat & final_dcorr) {
	int i, j, c, n;
	cv::Vec3d sum;
	
	std::vector<int> li, lj, lc;

	for (i = 0; i < final_corr.rows; i++) {
		for (j = 0; j < final_corr.cols; j++) {
			if (!final_corr.at<int>(i, j)) {
				if ((j > 0 && final_corr.at<int>(i, j-1) == 1) || (j < final_corr.cols - 1 && final_corr.at<int>(i, j+1) == 1) || (i > 0 && final_corr.at<int>(i-1, j) == 1) || (i < final_corr.rows - 1 && final_corr.at<int>(i+1, j) == 1)) {
					li.push_back(i);
					lj.push_back(j);
					lc.push_back(2);
					final_corr.at<int>(i, j) = 2;
				}
				else
					final_corr.at<int>(i, j) = INT_MAX;
			}
		}
	}

	for (unsigned int k = 0; k < li.size(); k++) {
		i = li[k];
		j = lj[k];
		c = lc[k];

		n = 0;
		sum = 0.0;
		if (j > 0) {
			if (final_corr.at<int>(i, j-1) < c) {
				sum += final_dcorr.at<cv::Vec3d>(i, j-1);
				n++;
			}
			else if (final_corr.at<int>(i, j-1) == INT_MAX) {
				li.push_back(i);
				lj.push_back(j-1);
				lc.push_back(c+1);
				final_corr.at<int>(i, j-1) = c+1;
			}
		}
		if (j < final_corr.cols - 1) {
			if (final_corr.at<int>(i, j+1) < c) {
				sum += final_dcorr.at<cv::Vec3d>(i, j+1);
				n++;
			}
			else if (final_corr.at<int>(i, j+1) == INT_MAX) {
				li.push_back(i);
				lj.push_back(j+1);
				lc.push_back(c+1);
				final_corr.at<int>(i, j+1) = c+1;
			}
		}
		if (i > 0) {
			if (final_corr.at<int>(i-1, j) < c) {
				sum += final_dcorr.at<cv::Vec3d>(i-1, j);
				n++;
			}
			else if (final_corr.at<int>(i-1, j) == INT_MAX) {
				li.push_back(i-1);
				lj.push_back(j);
				lc.push_back(c+1);
				final_corr.at<int>(i-1, j) = c+1;
			}
		}
		if (i < final_corr.rows - 1) {
			if (final_corr.at<int>(i+1, j) < c) {
				sum += final_dcorr.at<cv::Vec3d>(i+1, j);
				n++;
			}
			else if (final_corr.at<int>(i+1, j) == INT_MAX) {
				li.push_back(i+1);
				lj.push_back(j);
				lc.push_back(c+1);
				final_corr.at<int>(i+1, j) = c+1;
			}
		}

		final_dcorr.at<cv::Vec3d>(i, j) = sum/n;
	}
}

void Aligner::transpose_matrix(double matrix[3][3], double matrix_T[3][3]) const {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			matrix_T[i][j] = matrix[j][i];
}

void Aligner::compute_depth_map(const std::vector<cv::Point3d> & cloud) {
	double matrix[3][3];
	cv::Point3d tmp_point, translation_vector(transf[0], transf[1], transf[2]);
	compute_rotation_matrix(matrix, transf[3], transf[4], transf[5]);
	img.create(480, 640, CV_8U);
	img.setTo(0);
	cv::Mat final_dcorr(480, 640, CV_64FC3);
	final_dcorr.setTo(DBL_MAX);
	cv::Mat final_corr(480, 640, CV_32S);
	final_corr.setTo(0);
	for (int i = 0, len = cloud.size(); i < len; i++) {
		tmp_point = transform_point(matrix, translation_vector, cloud[i]);
		row = cvRound(-tmp_point.y/MODEL_RESOLUTION + final_proj_origin.y);
		col = cvRound(tmp_point.x/MODEL_RESOLUTION + final_proj_origin.x);
		if (row >= 0 && col >= 0 && row < final_dcorr.rows && col < final_dcorr.cols) {
			if (row < avg.rows && col < avg.cols) {
				double d = fabs(tmp_point.z - avg.at<cv::Vec3d>(row,col)[2]);
				if (d < OUTLIER_THRESHOLD) {
					final_dcorr.at<cv::Vec3d>(row, col) = tmp_point;
					final_corr.at<int>(row, col) = 1;
				}
			}
			else {
				final_dcorr.at<cv::Vec3d>(row, col) = tmp_point;
				final_corr.at<int>(row, col) = 1;
			}
		}
	}

	hole_filling(final_corr, final_dcorr);

	for (int i = 0; i < final_dcorr.rows; i++) {
		for (int j = 0; j < final_dcorr.cols; j++) {
		    if (final_corr.at<int>(i, j)) {
    			double z = 127.0 + 3.0*final_dcorr.at<cv::Vec3d>(i, j)[2];
	    		img.at<uchar>(i, j) = (z < 0.0 ? 0 : (z > 255.0 ? 255 : z));
    		}
		}
	}

}

void Aligner::compute_registered_depth_map(const std::vector<cv::Point3d> & cloud, const std::vector<cv::Point2d> & pixels) {
    // compute alignment tranformation
	double matrix[3][3];
	cv::Point3d tmp_point, translation_vector(transf[0], transf[1], transf[2]);
	compute_rotation_matrix(matrix, transf[3], transf[4], transf[5]);
	
	// create images
	img.create(480, 640, CV_8U);
	img.setTo(0);
	cv::Mat final_dcorr(480, 640, CV_64FC3);
	final_dcorr.setTo(DBL_MAX);
	cv::Mat final_corr(480, 640, CV_32S);
	final_corr.setTo(0);
	
	// find proper points
	for (int i = 0, len = pixels.size(); i < len; i++) {
        // deproject point
        cv::Point3d tmp_point = transform_point(matrix, translation_vector, cloud[i]);
		row = cvRound(-tmp_point.y/MODEL_RESOLUTION + final_proj_origin.y);
		col = cvRound(tmp_point.x/MODEL_RESOLUTION + final_proj_origin.x);
        
        // texture coordinates
        const int text_row = pixels[i].x;
        const int text_col = pixels[i].y;
        
        // draw accordingly
        if (row >= 0 && col >= 0 && row < img.rows && col < img.cols) {
	        final_dcorr.at<cv::Vec3d>(text_row, text_col) = tmp_point;
			final_corr.at<int>(text_row, text_col) = 1;
        }
    }

	hole_filling(final_corr, final_dcorr);

	// generate final image
	for (int i = 0; i < final_dcorr.rows; i++) {
		for (int j = 0; j < final_dcorr.cols; j++) {
		    if (final_corr.at<int>(i, j)) {
    			double z = 127.0 + 3.0*final_dcorr.at<cv::Vec3d>(i, j)[2];
	    		img.at<uchar>(i, j) = (z < 0.0 ? 0 : (z > 255.0 ? 255 : z));
    		}
		}
	}

}

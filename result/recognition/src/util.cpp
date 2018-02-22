#include "util.hpp"

void bbreg(Eigen::MatrixXd & boundingbox, Eigen::MatrixXd & reg) {
	assert(boundingbox.cols() == 5);
	assert(reg.cols() == 4);
    assert(boundingbox.rows() == reg.rows());

	int numOfBB = boundingbox.rows();
    Eigen::Matrix<double, Eigen::Dynamic, 1> w = boundingbox.col(2).cast<double>() - boundingbox.col(0).cast<double>() + Eigen::MatrixXd::Ones(numOfBB, 1);
	Eigen::Matrix<double, Eigen::Dynamic, 1> h = boundingbox.col(3).cast<double>() - boundingbox.col(1).cast<double>() + Eigen::MatrixXd::Ones(numOfBB, 1);
	boundingbox.col(0) += w.cwiseProduct(reg.col(0));
	boundingbox.col(1) += h.cwiseProduct(reg.col(1));
	boundingbox.col(2) += w.cwiseProduct(reg.col(2));
	boundingbox.col(3) += h.cwiseProduct(reg.col(3));
}

void pad(Eigen::MatrixXd &boundingbox, double w, double h, Eigen::MatrixXd &result) {
	assert(boundingbox.cols() == 5);

	int numOfBB = boundingbox.rows();
	result.resize(numOfBB, 10);

	Eigen::Matrix<double, Eigen::Dynamic, 1> tmpw = boundingbox.col(2).cast<double>() - boundingbox.col(0).cast<double>() + Eigen::MatrixXd::Ones(numOfBB, 1);
	Eigen::Matrix<double, Eigen::Dynamic, 1> tmph = boundingbox.col(3).cast<double>() - boundingbox.col(1).cast<double>() + Eigen::MatrixXd::Ones(numOfBB, 1);
	Eigen::MatrixXd dx = Eigen::MatrixXd::Ones(numOfBB, 1);
	Eigen::MatrixXd dy = Eigen::MatrixXd::Ones(numOfBB, 1);
	Eigen::Matrix<double, Eigen::Dynamic, 1> edx = tmpw.replicate(1, 1);
	Eigen::Matrix<double, Eigen::Dynamic, 1> edy = tmph.replicate(1, 1);

	auto x = Eigen::MatrixXd(boundingbox.col(0));
	auto y = Eigen::MatrixXd(boundingbox.col(1));
	auto ex = Eigen::MatrixXd(boundingbox.col(2));
	auto ey = Eigen::MatrixXd(boundingbox.col(3));

	Eigen::MatrixXd w_matrix;
	w_matrix.resize(ex.rows(), ex.cols());
	w_matrix.fill(w);
	Eigen::VectorXi tmp = _find(ex, w_matrix);

	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		edx(j) = -ex(j) + w - 1 + tmpw(j);
		ex(j) = w - 1;
	}

	Eigen::MatrixXd h_matrix;
	h_matrix.resize(ey.rows(), ey.cols());
	h_matrix.fill(h);
	tmp = _find(ey, h_matrix);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		edy(j) = -ey(j) + h - 1 + tmph(j);
		ey(j) = h - 1;
	}

	Eigen::MatrixXd one_matrix = Eigen::MatrixXd::Ones(x.rows(), x.cols());
	tmp = _find(one_matrix, x);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		dx(j) = 2 - x(j);
		x(j) = 1;
	}
	
	tmp = _find(one_matrix, y);
	for (int i = 0; i < tmp.size(); i++){
		int j = tmp(i);
		dy(j) = 2 - y(j);
		y(j) = 1;
	}
	dy -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	edy -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	dx -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	edx -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	y -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	ey -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	x -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	ex -= Eigen::MatrixXd::Ones(dy.rows(), dy.cols());
	
	result << dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;
}

void rerec(Eigen::MatrixXd &boundingbox) {
	assert(boundingbox.cols() == 5);
	
	auto w = Eigen::MatrixXd(boundingbox.col(2) - boundingbox.col(0));
	auto h = Eigen::MatrixXd(boundingbox.col(3) - boundingbox.col(1));
	auto l = w.cwiseMax(h);
	boundingbox.col(0) += w*0.5 - l*0.5;
	boundingbox.col(1) += h*0.5 - l*0.5;
	Eigen::MatrixXd ll;
	ll.resize(l.rows(), l.cols() * 2);
	ll << l, l;
	boundingbox.middleCols(2, 2) = boundingbox.middleCols(0, 2) + ll;
}

Eigen::VectorXi _find(Eigen::MatrixXd A, Eigen::MatrixXd B) {
    // find index where A > B
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> C = A.array() > B.array();
    Eigen::VectorXi I = Eigen::VectorXi::LinSpaced(C.size(), 0, C.size() - 1);
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&C](int i){return C(i); }) - I.data());
    return I;
}

Eigen::VectorXi _find(Eigen::MatrixXd A, double b) {
    Eigen::MatrixXd B = Eigen::MatrixXd(A.rows(), A.cols());
    B.fill(b);
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> C = A.array() > B.array();
    Eigen::VectorXi I = Eigen::VectorXi::LinSpaced(C.size(), 0, C.size() - 1);
    I.conservativeResize(std::stable_partition(I.data(), I.data() + I.size(), [&C](int i){return C(i); }) - I.data());
    return I;
}

void _find(std::vector<double> & A, double b, std::vector<int> & C) {
    for (int i = 0; i < A.size(); i++){
        if (A.at(i) > b){
            C.push_back(i);
        }
    }
}

void _fix(Eigen::MatrixXd & M) {
    for (int i = 0; i < M.cols(); i++) {
        for (int j = 0; j < M.rows(); j++) {

            int temp = (int)M(j, i);

            if (temp > M(j, i)) temp--;
            else if (M(j, i) - temp > 0.9) temp++;

            M(j, i) = (double)temp;
        }
    }
}

Eigen::MatrixXd subOneRow(Eigen::MatrixXd M, int index) {
    assert(M.rows() > index);
    Eigen::MatrixXd out(M.rows() - 1, M.cols());
    for (int i = 0, j = 0; i < M.rows(), j < out.rows(); ){
        if (i != index){
            out.row(j) = M.row(i);
            i++;
            j++;
        }
        else
            i++;
    }
    return out;
}

Eigen::MatrixXd subOneRowRerange(Eigen::MatrixXd & M, std::vector<int> & I) {
    Eigen::MatrixXd out(I.size() - 1, M.cols());
    for (int i = 0; i < I.size() - 1; i++){
        out.row(i) = M.row(I[i]);
    }
    return out;
}

void npwhere_vec(std::vector<int> & index, const std::vector<double> & value, const double threshold) {
    std::vector<int> out;
    auto i = index.begin();
    auto j = value.begin();
    for (; i != index.end(), j != value.end(); i++, j++){
        if (*j <= threshold){
            out.push_back(*i);
        }
    }
    index.resize(out.size());
    index = out;
}

void _select(Eigen::MatrixXd & src, Eigen::MatrixXd & dst, const std::vector<int> & pick) {
    Eigen::MatrixXd _src = src.replicate(1,1);
    int new_height = pick.size();
    int new_width = src.cols();
    dst.resize(new_height, new_width);
    for(int i=0; i < pick.size(); i++){
        dst.row(i) = _src.row(pick[i]);
    }
}

void convertToMatrix(caffe::Blob<float> * prob, caffe::Blob<float> * conv, Eigen::MatrixXd & map, std::vector<Eigen::MatrixXd> & reg) {
    int height = prob->height();
    int width = prob->width();

    // convert to map
    float* data = prob->mutable_cpu_data() + height * width;
    cv::Mat prob_mat(height, width, CV_32FC1, data);
    cv2eigen(prob_mat, map);

    // convert to reg
    data = conv->mutable_cpu_data();
    Eigen::MatrixXd eachReg;
    eachReg.resize(height, width);
    for(int i=0; i < conv->channels(); i++){
        cv::Mat reg_mat(height, width, CV_32FC1, data);
        cv2eigen(reg_mat, eachReg);
        reg.push_back(eachReg);
        data += height * width;
    }
}

void convertToVector(caffe::Blob<float> * prob, std::vector<double> & score) {

    assert(prob->channels() == 2);
    int num = prob->num();

    // convert to score
    float* data = prob->mutable_cpu_data();
    data++;
    for (int i = 0; i < num; i++){
        //std::cout << *data << std::endl;
        score.push_back(*data);
        data += 2;
    }
}

void filter(Eigen::MatrixXd & total_boxes, Eigen::VectorXi & pass_t, Eigen::MatrixXd & score) {
    Eigen::MatrixXd new_boxes;
    new_boxes.resize(pass_t.size(), 5);
    for (int i = 0; i < pass_t.size(); i++){
        Eigen::MatrixXd tmp;
        tmp.resize(1, 5);
        tmp << total_boxes(pass_t(i), 0), total_boxes(pass_t(i), 1), total_boxes(pass_t(i), 2), total_boxes(pass_t(i), 3), score(pass_t(i));
        new_boxes.row(i) = tmp;
    }
    total_boxes.resize(pass_t.size(), 5);
    total_boxes << new_boxes;
}

void filter(Eigen::MatrixXd & total_boxes, std::vector<int> & pass_t, std::vector<double> & score) {
    Eigen::MatrixXd new_boxes;
    new_boxes.resize(pass_t.size(), 5);
    for (int i = 0; i < pass_t.size(); i++){
        Eigen::MatrixXd tmp;
        tmp.resize(1, 5);
        tmp << total_boxes(pass_t.at(i), 0), total_boxes(pass_t.at(i), 1), total_boxes(pass_t.at(i), 2), total_boxes(pass_t.at(i), 3), score.at(pass_t.at(i));
        new_boxes.row(i) = tmp;
    }
    total_boxes.resize(pass_t.size(), 5);
    total_boxes << new_boxes;
}

void getMV(caffe::Blob<float> * conv, Eigen::MatrixXd & mv, std::vector<int> & pass_t) {
    int num = conv->num();
    int channels = conv->channels();

    // convert to Eigen::MatrixXd
    Eigen::MatrixXd conv_m;
    float* data = conv->mutable_cpu_data();
    cv::Mat conv_mat(num, channels, CV_32FC1, data);
    cv2eigen(conv_mat, conv_m);
    _select(conv_m, mv, pass_t);
}

void nms(Eigen::MatrixXd & boundingbox, float threshold, const std::string & type, std::vector<int> & pick) {
    assert(boundingbox.cols() == 5 || boundingbox.cols() == 9);
    if (boundingbox.rows() < 1) return;

    Eigen::MatrixXd x1 = Eigen::MatrixXd(boundingbox.col(0));
    Eigen::MatrixXd y1 = Eigen::MatrixXd(boundingbox.col(1));
    Eigen::MatrixXd x2 = Eigen::MatrixXd(boundingbox.col(2));
    Eigen::MatrixXd y2 = Eigen::MatrixXd(boundingbox.col(3));
    Eigen::MatrixXd s = Eigen::MatrixXd(boundingbox.col(4));
    Eigen::MatrixXd one_vector = Eigen::MatrixXd::Ones(x1.rows(), 1);
    Eigen::MatrixXd area = (x2 - x1 + one_vector).cwiseProduct(y2 - y1 + one_vector);
    Eigen::MatrixXd _vals;
    Eigen::MatrixXi _I;
    igl::sort(s, 1, true, _vals, _I);
    std::vector<int> I(_I.data(), _I.data() + _I.rows()*_I.cols());

    while (I.size() > 0){
        //xx1 = max(x1(i), x1(I(1:last-1)));
        Eigen::MatrixXd x1_powerful = Eigen::MatrixXd(I.size() - 1, 1);
        x1_powerful.fill(x1(I.back()));
        Eigen::MatrixXd xx1 = x1_powerful.cwiseMax(subOneRowRerange(x1, I));

        Eigen::MatrixXd y1_powerful = Eigen::MatrixXd(I.size() - 1, 1);
        y1_powerful.fill(y1(I.back()));
        Eigen::MatrixXd yy1 = y1_powerful.cwiseMax(subOneRowRerange(y1, I));

        Eigen::MatrixXd x2_powerful = Eigen::MatrixXd(I.size() - 1, 1);
        x2_powerful.fill(x2(I.back()));
        Eigen::MatrixXd xx2 = x2_powerful.cwiseMin(subOneRowRerange(x2, I));

        Eigen::MatrixXd y2_powerful = Eigen::MatrixXd(I.size() - 1, 1);
        y2_powerful.fill(y2(I.back()));
        Eigen::MatrixXd yy2 = y2_powerful.cwiseMin(subOneRowRerange(y2, I));

        auto w = Eigen::MatrixXd::Zero(I.size() - 1, 1).cwiseMax(xx2-xx1+Eigen::MatrixXd::Ones(I.size()-1,1));
        auto h = Eigen::MatrixXd::Zero(I.size() - 1, 1).cwiseMax(yy2-yy1+Eigen::MatrixXd::Ones(I.size()-1,1));
        auto inter = w.cwiseProduct(h);

        Eigen::MatrixXd o;
        Eigen::MatrixXd area_powerful = Eigen::MatrixXd(I.size() - 1, 1);
        area_powerful.fill(area(I.back()));
        if (type == "Min"){
            o = inter.cwiseQuotient(area_powerful.cwiseMin(subOneRowRerange(area, I)));
        }
        else{
            Eigen::MatrixXd tmp = area_powerful + subOneRowRerange(area, I) - inter;
            o = inter.cwiseQuotient(tmp);
        }

        pick.push_back(I.back());

        std::vector<double> o_list(o.data(), o.data() + o.rows()*o.cols());
        npwhere_vec(I, o_list, threshold);
    }
}

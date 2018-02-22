#include <normalize.hpp>

cv::Mat align_eyes(const cv::Mat & img, cv::Point & left_eye_center, cv::Point & right_eye_center, cv::Point & mouth_center) {
    // rotate matrix computation
    const double angle = atan2(right_eye_center.y - left_eye_center.y, right_eye_center.x - left_eye_center.x);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2), angle * 180.0 / CV_PI, 1.0);
    
    // rotate image
    cv::Mat ret;
    cv::warpAffine(img, ret, rotation_matrix, img.size());
    
    // rotate landmarks
    left_eye_center = cv::Point((left_eye_center.x - img.cols / 2)*cos(-angle) - (left_eye_center.y - img.rows / 2)*sin(-angle) + img.cols / 2, (left_eye_center.x - img.cols / 2)*sin(-angle) + (left_eye_center.y - img.rows / 2)*cos(-angle) + img.rows / 2);
    right_eye_center = cv::Point((right_eye_center.x - img.cols / 2)*cos(-angle) - (right_eye_center.y - img.rows / 2)*sin(-angle) + img.cols / 2, (right_eye_center.x - img.cols / 2)*sin(-angle) + (right_eye_center.y - img.rows / 2)*cos(-angle) + img.rows / 2);
    mouth_center = cv::Point((mouth_center.x - img.cols / 2)*cos(-angle) - (mouth_center.y - img.rows / 2)*sin(-angle) + img.cols / 2, (mouth_center.x - img.cols / 2)*sin(-angle) + (mouth_center.y - img.rows / 2)*cos(-angle) + img.rows / 2);
    return ret;
}

cv::Mat resize(const cv::Mat & img, cv::Point & eyes_midpoint, cv::Point & mouth_center) {
    const double ratio = EC_MC_Y / cv::norm(eyes_midpoint - mouth_center);
    cv::Mat ret;
    cv::resize(img, ret, cv::Size(img.cols * ratio, img.rows * ratio));
    eyes_midpoint *= ratio;
    mouth_center *= ratio;
    return ret;
}

cv::Mat crop(const cv::Mat & img, cv::Point & eyes_midpoint, cv::Point & mouth_center) {
    cv::Rect roi(eyes_midpoint.x - IMG_SIZE / 2, eyes_midpoint.y - EC_Y, IMG_SIZE, IMG_SIZE);
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width >= img.cols || roi.y + roi.height >= img.rows) {
        cv::Mat bordered;
        cv::copyMakeBorder(img, bordered, std::abs(std::min(roi.y, 0)), roi.y + roi.height, std::abs(std::min(roi.x, 0)), roi.x + roi.width, cv::BORDER_REPLICATE);
        roi.x = std::max(roi.x, 0);
        roi.y = std::max(roi.y, 0);
        return bordered(roi);
    }
    return img(roi);
}

cv::Mat normalize(const cv::Mat & img, const std::vector<cv::Point> & landmarks) {
    // retrieve landmark centers
    cv::Point left_eye_center = landmarks[0];
    cv::Point right_eye_center = landmarks[1];
    cv::Point mouth_center = (landmarks[3] + landmarks[4]) / 2;
    
    cv::Mat aligned = align_eyes(img, left_eye_center, right_eye_center, mouth_center);
    cv::Point eye_midpoint = (left_eye_center + right_eye_center) / 2;
    cv::Mat resized = resize(aligned, eye_midpoint, mouth_center);
    cv::Mat ret = crop(resized, eye_midpoint, mouth_center);
    
    // convert to gray scale
    // cvtColor(ret, ret, CV_BGR2GRAY);
    return ret;
}

cv::Mat normalize_depth(Aligner *aligner) {
    // retrieve landmark centers
    cv::Point left_eye_center = aligner->landmarks[0];
    cv::Point right_eye_center = aligner->landmarks[1];;
    cv::Point mouth_center = (aligner->landmarks[3]+aligner->landmarks[4])/2;
    
    cv::Mat aligned = align_eyes(aligner->img, left_eye_center, right_eye_center, mouth_center);
    cv::Point eye_midpoint = (left_eye_center + right_eye_center) / 2;
    cv::Mat resized = resize(aligned, eye_midpoint, mouth_center);
    cv::Mat ret = crop(resized, eye_midpoint, mouth_center);
    
    // convert to gray scale
    // cvtColor(ret, ret, CV_BGR2GRAY);
    return ret;
}

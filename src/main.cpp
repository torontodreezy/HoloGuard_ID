#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <torch/torch.h>
using namespace std;


int main() {
    cv::Mat img = cv::Mat::zeros(1000, 1000, CV_8UC3);

    cv::imshow("1", img);
    cv::waitKey(0);

}
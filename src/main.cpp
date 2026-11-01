#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <torch/torch.h>
using namespace std;


int main() {
    cv::Mat photo = cv::imread("/home/bladerunner/CLionProjects/engines/dataset/images/origins/ID/id01_04_02/img_0024.jpg");
    cv::imshow("1", photo);
    cv::waitKey(0);

}
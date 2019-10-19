#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using namespace cv;
using namespace ml;

int main()
{
    // 准备测试图片
    Mat testImage = imread("../../data/mnist_test_img/0.0.jpg",0);
    resize(testImage,testImage,Size(20,20));
    Mat pImage = testImage.reshape(1,1);
    pImage.convertTo(pImage, CV_32FC1);

    // 导入SVM模型
    string modelpath = "../../model/digits.xml";
    Ptr<SVM> model = SVM::load(modelpath);

    //预测
    int preclass = (int) model->predict(pImage);
    cout << "testImage = " << preclass << endl;

    return 0;
}

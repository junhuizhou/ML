#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using namespace cv;
using namespace ml;

Mat detectHog(Mat ipImage)
{
    HOGDescriptor hog(Size(50,50),Size(10,10),Size(5,5),Size(5,5),3);
    Mat feactureMat;
    vector<float> descriptors;
    hog.compute(ipImage,descriptors);
    feactureMat = Mat::zeros(1,descriptors.size(),CV_32FC1);
    for(int i=0; i<descriptors.size(); i++)
    {
        feactureMat.at<float>(0,i) = descriptors[i];
    }
    return feactureMat;
}

int main()
{
    // 准备测试图片
    Mat testImage = imread("../svm_img/0/5.jpg",0);
    resize(testImage,testImage,Size(50,50));
    bool HogorNor = true;
    Mat pImage;
    if(HogorNor == true)
    {
        pImage = detectHog(testImage);
        cout << "use hog feacture" << endl;
    }
    else
    {
        pImage = testImage.reshape(1,1);
        cout << "use full image" << endl;
    }
    pImage.convertTo(pImage, CV_32FC1);

    // 导入SVM模型
    string modelpath = "../model/svm_rm.xml";
    Ptr<SVM> model = SVM::load(modelpath);

    //预测
    int preclass = (int) model->predict(pImage);
    cout << "testImage = " << preclass << endl;

    return 0;
}

#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using std::cout;
using std::endl;
using std::string;
using namespace cv;
using namespace cv::dnn;

void getMaxClass(const Mat &probBlob, int *classId, double *classProb);

int main()
{
    // 生成神经网络Net
    string lenetTxt = "../model/lenet_RM_test.prototxt";
    string caffeModel = "../model/lenet_iter_10000.caffemodel";
    Net net;
    try
    {
        net = readNetFromCaffe(lenetTxt,caffeModel);
    }
    catch(Exception &ee)
    {
        cout << "Exception: " << ee.what() << endl;
        if(net.empty())
        {
            cout << "Cant't load the network by using the flowing files:" << endl;
            cout << "lenetTxt: " << lenetTxt << endl;
            cout << "caffeModel: " << caffeModel << endl;
            exit(-1);
        }
    }

    // 输入测试样例img，并转化为net可以使用的blob格式
    Mat img = imread("../mnist_test_img/0.0.jpg",0);
    Mat inputBlob = blobFromImage(img,0.00390625f,Size(28,28),Scalar(),false);
    net.setInput(inputBlob,"data");
    
    // 预测即计算img属于各类型的概率
    Mat predict;
    predict = net.forward("prob");
    // cout << predict << endl;

    // 输出img最有可能属于的类型
    int classId;
    double classProb;
    getMaxClass(predict, &classId, &classProb);
    cout << "Best Class: " << classId << endl;
    cout << "Probability: " << classProb*100 << "%" << endl;

    return 0;
}

void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1,1);
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}
#include <iostream>
#include <cstring>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using namespace cv;
using namespace ml;

vector<Mat> trainingImages;
vector<int> trainingLabels;

int openfile(int flag, const char * dpath);

int main()
{
    // 准备训练数据和标签
    Mat classes;
    openfile(1,"../../data/1");
    openfile(0,"../../data/0");
    Mat trainingData(trainingImages.size(),trainingImages[0].cols,CV_32FC1);
    for (int i=0; i<trainingImages.size(); i++)
    {
        Mat tmp(trainingImages[i]);
        tmp.copyTo(trainingData.row(i));
    }
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);
    classes.convertTo(classes, CV_32SC1);

    // 配置SVM参数
    Ptr<SVM> model = SVM::create();
    model->setType(SVM::C_SVC);
    model->setKernel(SVM::LINEAR);
    model->setGamma(1);
    model->setC(1);
    model->setCoef0(0);
    model->setNu(0);
    model->setP(0);
    model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER,1000,0.01));

    // 训练并保存模型
    Ptr<TrainData> tData = TrainData::create(trainingData, ROW_SAMPLE, classes);
    model->train(tData);
    model->save("../../model/digits.xml");

    return 0;
}

int openfile(int flag, const char * dpath)
{
    // 获取图像路径
    vector<string> files;
    DIR *dir;
    struct dirent *ptr;
    if((dir = opendir(dpath)) == NULL)  //打开文件夹
    {
        return -1;
    }
    while((ptr = readdir(dir)) != NULL)
    {
        if(ptr->d_type > 4)
        {
            char name[30];
            char path[200];
            sprintf(path, "%s/%s", dpath, ptr->d_name);
            files.push_back(path);
            puts(path);
        }
    }
    closedir(dir);

    // 将图像对应的值存放到容器中
    for(int i=0; i<files.size(); i++)
    {
        Mat img = imread(files[i].c_str(),0);
        Mat line_i = img.reshape(1,1);
        trainingImages.push_back(line_i);
        trainingLabels.push_back(flag);
    }
}
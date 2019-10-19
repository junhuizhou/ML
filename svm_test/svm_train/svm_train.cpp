#include <iostream>
#include <cstring>
#include <vector>
#include <time.h>
#include <dirent.h>
#include <sys/types.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace cv;
using namespace cv::ml;

vector<Mat> trainingImages;
vector<int> trainingLabels;

int openfile(int flag, const char * dpath);

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
    // 读取图片并打标签，两个即二分类，多个即多分类
    openfile(0,"../../svm_img/0");
    openfile(1,"../../svm_img/1");
    openfile(2,"../../svm_img/2");
    openfile(3,"../../svm_img/3");
    openfile(4,"../../svm_img/4");
    openfile(5,"../../svm_img/5");

    // 训练数据和标签的格式转化
    Mat classes;
    Mat trainingData(trainingImages.size(),trainingImages[0].cols,CV_32FC1);
    for(int i=0; i<trainingImages.size(); i++)
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
    model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER,1000,0.01));
    
    // 训练并保存模型
    Ptr<TrainData> tData = TrainData::create(trainingData, ROW_SAMPLE, classes);
    enum {autoTrain, manualTrain};
    bool trainMethod = true;
    if(trainMethod == autoTrain)
    {
        cout << "train by auto method" << endl;
        model->trainAuto(tData);
    }
    else
    {
        cout << "train by manual method" << endl;
        model->setGamma(1);
        model->setC(1);
        model->setCoef0(0);
        model->setNu(0);
        model->setP(0);
        model->train(tData);
    }
    model->save("../../model/svm_rm.xml");
    
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
    bool HogorNot = true;
    for(int i=0; i<files.size(); i++)
    {
        Mat img = imread(files[i].c_str(),0);
        if(HogorNot == false)
        {
            Mat line_i = img.reshape(1,1);
            trainingImages.push_back(line_i);
            trainingLabels.push_back(flag);
        }
        else
        {
            equalizeHist(img,img);
            Mat hogFeactureMat = detectHog(img);
            if(!hogFeactureMat.empty())
            {
                trainingImages.push_back(hogFeactureMat);
                trainingLabels.push_back(flag);
            }
        }
        
        
    }
}

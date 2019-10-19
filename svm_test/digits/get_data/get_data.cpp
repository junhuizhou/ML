#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::sprintf;
using std::cout;
using std::endl;

int main()
{
    char ad[128]={0};
    int filename = 0, filenum = 0;
    Mat img = imread("../../digits.png",0);
    int b = 20;
    int m = img.rows / b;
    int n = img.cols / b;

    for (int i=0; i<m; i++)
    {
        int offsetRow = i*b;
        if(i%5==0 && i!=0)
        {
            ++filename;
            filenum = 0;
        }
        for (int j=0; j<n; j++)
        {
            int offsetCol = j*b;
            sprintf(ad,"../../data/%d/%d.jpg",filename,filenum++);
            Mat tmp;
            img(Range(offsetRow,offsetRow+b),Range(offsetCol,offsetCol+b)).copyTo(tmp);
            imwrite(ad,tmp);
        }
    }
    return 0;
}
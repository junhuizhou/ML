# SVM_TEST

## 文件树

```
svm_test
├── build                       //存放svm_test.out
├── CMakeLists.txt
├── digits                      //SVM测试mnist，一个demo
│   ├── data                    //存放mnist图片
│   │   └── mnist_test_img
│   ├── digits.png
│   ├── get_data                //由digits.png得到mnist数据存到data中
│   │   ├── CMakeLists.txt
│   │   └── get_data.cpp
│   ├── model                   //train_data训练得到的模型.xml
│   │   └── digits.xml
│   ├── test_data               //调用model中的模型做分类
│   │   ├── build
│   │   ├── CMakeLists.txt
│   │   └── test_data.cpp
│   └── train_data              //训练data中的数据得到model
│       ├── build
│       ├── CMakeLists.txt
│       └── train_data.cpp
├── model                       //训练装甲板贴纸得到的.xml模型
├── readme.md
├── svm_img                     //RM中装甲板贴纸
├── svm_test.cpp                //调用model中模型分类装甲板
├── svm_train                   //训练RM中的装甲板贴纸数据
    ├── build
    ├── CMakeLists.txt
    └── svm_train.cpp
```
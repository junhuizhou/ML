### 使用方式一：caffe命令
* 在caffe_test路径下打开终端，运行

../../caffe/build/examples/cpp_classification/classification.bin ./model/lenet_RM_test.prototxt ./model/lenet_iter_10000.caffemodel ./model/mean.binaryproto ./model/mnistLabel.txt mnist_test_img/0.0.jpg

### 使用方式二：OpenCV调用
* 更改CMakeLists.txt中的OpenCV版本
* 在build中打开终端运行cmake .
* 再运行make
* 最后运行./caffe_test
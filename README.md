## dlex-cnn : Deep Learning Exercise
A simple convolution neural network framework for exercise.

---

## Features
* Lightweight, pure c++ and cuda implemented.
* Easy to understand, and the algorithm implementation details are presented as annotations.

---

## Basic Modules
![](./res/readme_image/basic-module.png "Basic modules")

## Experiment
lenet-mnist
![](./res/readme_image/lenet-mnist.png "lenet-mnist")

## Build
Windows: Just open the project file with VS2013 or VS2015, and compile it.

Linux: 
Build libdlex_cnn.so
```
cd linux/cmake
cmake ..
make
```

Build demo
```
cd linux/dlex_cnn_test/cmake
cmake ..
make 
```

### License
MIT

### Reference
* Caffe: https://github.com/BVLC/caffe
* MxNet: https://github.com/apache/incubator-mxnet
* tiny-dnn: https://github.com/tiny-dnn/tiny-dnn
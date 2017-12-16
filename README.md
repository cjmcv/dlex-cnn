## dlex-cnn : Deep Learning Exercise
A simple convolution neural network framework.

---

## Features
* Needn't rely on any third-party libraries, pure c++ and cuda implemented.
* Easy to understand, and the algorithm implementation details are presented as annotations.

---

## Basic Module
* Network: The biggest unit, which contains a Graph, is in charge of network's training, inference and model's I/O etc.
* Graph: Organizes all of the nodes in a network.
* Node: Mainly holds operators and tensor data.
* Tensor: The basic data structure.
* Operator: Contains activation(ReLU/Sigmoid/Tanh), convolution, deconvolution, inner product, pooling(max/ave), softmax and softmax loss for now.
* Optimizer: Only support SGD for now.
* Model's I/O: To save and load models.
* Prefetcher: Prefetch data for CUDA training and inference.
* Trainer: Provides an easy way to create a new network, and it also includes some typical network structures.
* Operators factory: A simple factory for operators.
* Thread Pool: Takes a part in prefetching and acceleration, which needs multi-threads to assist.
* Memory Pool. [To Do]


## Installation
* Windows: Only contains VS2013 project files with x64 in the "windows" folder. And of course, this program is pure c++ and cuda implemented, you can easily create project files in other IDE.
* Linux: There are two CMakeLists.txt in "linux" folder to generate the target files you want. More details in "cmake/README.md".


### License
MIT


### Reference
* Caffe: https://github.com/BVLC/caffe
* MxNet: https://github.com/apache/incubator-mxnet
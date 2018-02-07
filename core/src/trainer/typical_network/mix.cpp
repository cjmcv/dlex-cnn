////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "trainer/typical_network.h"

namespace dlex_cnn {
template <typename Dtype>
int TypicalNet::mix(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network) {
  printf("start building net\n");
  NetCreator<Dtype> creator;

  InputOpParam input_params;
  input_params.num = num;
  input_params.channels = channels;
  input_params.height = height;
  input_params.width = width;
  std::string input_name = "input";
  creator.CreateInputNode(input_name, input_params, network);

  std::string conv1_params = "blas_enable:1, kernel_num:20,  kernel_h:5, kernel_w:5, stride_h:1, stride_w:1,  pad_h:0, pad_w:0, dilation_h:1, dilation_w:1,";
  std::string conv1_name = "conv1";
  creator.CreateConvNode(input_name, conv1_name, conv1_params, network);

  std::string pool1_params = "pooling_type:0,kernel_h:2,kernel_w:2,stride_h:2,stride_w:2,pad_h:0,pad_w:0,global_pooling:0,";
  std::string pool1_name = "pool1";
  creator.CreatePoolNode(conv1_name, pool1_name, pool1_params, network);

  std::string deconv2_params = "blas_enable:1, kernel_channels:6, kernel_h:3,kernel_w:3, stride_h:1,stride_w:1, pad_h:1,pad_w:1, dilation_h:1,dilation_w:1,";
  std::string deconv2_name = "deconv2";
  creator.CreateDeconvNode(pool1_name, deconv2_name, deconv2_params, network);

  std::string pool2_params = "pooling_type:0,kernel_h:2,kernel_w:2,stride_h:2,stride_w:2,pad_h:0,pad_w:0,global_pooling:0,";
  std::string pool2_name = "pool2";
  creator.CreatePoolNode(deconv2_name, pool2_name, pool2_params, network);

  std::string fc1_params = "blas_enable:1, num_hidden:500,";
  std::string fc1_name = "fc1";
  creator.CreateInnerProductNode(pool2_name, fc1_name, fc1_params, network);

  std::string act1_params = "activation_type:0, negative_slope:0, ";
  std::string act1_name = "act1";
  creator.CreateActivationNode(fc1_name, act1_name, act1_params, network);

  std::string fc2_params = "blas_enable:1, num_hidden:10,";
  std::string fc2_name = "fc2";
  creator.CreateInnerProductNode(act1_name, fc2_name, fc2_params, network);

  std::string sm_params = ",";
  std::string sm_name = "sm";
  creator.CreateSoftmaxLossNode(fc2_name, sm_name, sm_params, network);

  std::string out_params = "label_dim:1,";
  std::string out_name = "output";
  creator.CreateOutputNode(sm_name, out_name, out_params, network);

  creator.CreateOptimizer("SGD", network);

  return 0;
}

template int TypicalNet::mix(const int num, const int channels, const int height, const int width, NetWork<float> &network);
template int TypicalNet::mix(const int num, const int channels, const int height, const int width, NetWork<double> &network);

}

////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "trainer/typical_network.h"

namespace dlex_cnn {
template <typename Dtype>
int TypicalNet::mlp(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network) {
  printf("start building net\n");
  NetCreator<Dtype> creator;

  InputOpParam input_params;
  input_params.num = num;
  input_params.channels = channels;
  input_params.height = height;
  input_params.width = width;
  std::string input_name = "input";
  creator.CreateInputNode(input_name, input_params, network);

  std::string fc0_params = "blas_enable:1, num_hidden:256,";
  std::string fc0_name = "fc0";
  creator.CreateInnerProductNode(input_name, fc0_name, fc0_params, network);

  std::string fc1_params = "blas_enable:1, num_hidden:256,";
  std::string fc1_name = "fc1";
  creator.CreateInnerProductNode(fc0_name, fc1_name, fc1_params, network);

  std::string fc2_params = "blas_enable:1, num_hidden:10,";
  std::string fc2_name = "fc2";
  creator.CreateInnerProductNode(fc1_name, fc2_name, fc2_params, network);

  std::string sm_params = ",";
  std::string sm_name = "sm";
  creator.CreateSoftmaxLossNode(fc2_name, sm_name, sm_params, network);

  std::string out_params = "label_dim:1,";
  std::string out_name = "output";
  creator.CreateOutputNode(sm_name, out_name, out_params, network);

  creator.CreateOptimizer("SGD", network);

  return 0;
}

template int TypicalNet::mlp(const int num, const int channels, const int height, const int width, NetWork<float> &network);
template int TypicalNet::mlp(const int num, const int channels, const int height, const int width, NetWork<double> &network);


}

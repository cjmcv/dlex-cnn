////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_NET_CREATOR_HPP_
#define DLEX_NET_CREATOR_HPP_
//
#include <iostream>

#include "network.h"
// operator
#include "operator/input_op.h"
#include "operator/output_op.h"
#include "operator/inner_product_op.h"
#include "operator/convolution_op.h"
#include "operator/deconvolution_op.h"
#include "operator/pooling_op.h"
#include "operator/activation_simple_op.h"
#include "operator/softmax_op.h"
#include "operator/operator_base.h"
#include "operator/operator_hybrid.h"
#include "operator/cross_entropy_lop.h"
#include "operator/softmax_cross_entropy_hop.h"

namespace dlex_cnn {
template <typename Dtype>
class NetCreator {
public:
  NetCreator() {};
  virtual ~NetCreator() {};

public:
  // Input
  int CreateInputNode(std::string node_name, std::string param, NetWork<Dtype> &network);
  int CreateInputNode(std::string node_name, InputOpParam param, NetWork<Dtype> &network);

  // Inner Product
  int CreateInnerProductNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateInnerProductNode(std::string in_node, std::string name, InnerProductOpParam param, NetWork<Dtype> &network);

  // Convolution
  int CreateConvNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateConvNode(std::string in_node, std::string name, ConvolutionOpParam param, NetWork<Dtype> &network);

  // Deconvolution
  int CreateDeconvNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateDeconvNode(std::string in_node, std::string name, DeconvolutionOpParam param, NetWork<Dtype> &network);

  // Activation
  int CreateActivationNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateActivationNode(std::string in_node, std::string name, ActivationOpParam param, NetWork<Dtype> &network);

  // Pooling
  int CreatePoolNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreatePoolNode(std::string in_node, std::string name, PoolingOpParam param, NetWork<Dtype> &network);

  // Softamx Cross Entropy Loss
  int CreateSoftmaxLossNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateSoftmaxLossNode(std::string in_node, std::string name, SoftmaxCrossEntropyLossHOpParam param, NetWork<Dtype> &network);

  // Output
  int CreateOutputNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network);
  int CreateOutputNode(std::string in_node, std::string name, OutputOpParam param, NetWork<Dtype> &network);

  // Optimizer
  int CreateOptimizer(std::string opt_type, NetWork<Dtype> &network);
};
}

#endif
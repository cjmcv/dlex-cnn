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

namespace dlex_cnn
{
	template <typename Dtype>
	class NetCreator
	{
	public:
		NetCreator() {};
		virtual ~NetCreator() {};

	public:
		// Input
		int createInputNode(std::string nodeName, std::string paramStr, NetWork<Dtype> &network);
		int createInputNode(std::string nodeName, InputOpParam param, NetWork<Dtype> &network);
		
		// Inner Product
		int createInnerProductNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createInnerProductNode(std::string inNode, std::string name, InnerProductOpParam param, NetWork<Dtype> &network);
		
		// Convolution
		int createConvNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createConvNode(std::string inNode, std::string name, ConvolutionOpParam param, NetWork<Dtype> &network);

		// Activation
		int createActivationNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createActivationNode(std::string inNode, std::string name, ActivationOpParam param, NetWork<Dtype> &network);

		// Pooling
		int createPoolNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createPoolNode(std::string inNode, std::string name, PoolingOpParam param, NetWork<Dtype> &network);

		// Softamx Cross Entropy Loss
		int createSoftmaxLossNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createSoftmaxLossNode(std::string inNode, std::string name, SoftmaxCrossEntropyLossHOpParam param, NetWork<Dtype> &network);

		// Output
		int createOutputNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network);
		int createOutputNode(std::string inNode, std::string name, OutputOpParam param, NetWork<Dtype> &network);
	};
}

#endif
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DLEX_HPP_
#define DLEX_DLEX_HPP_

//configure
#include "configure.h"
#include "util/math_functions.h"
#include "util/op_factory.h"
#include "util/thread_pool.h"

//operator
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

//network
#include "network.h"
//dataType define
#include "node.h"
#include "tensor.h"

//
#include "trainer/typical_network.h"

static void registerOpClass()
{	//register
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Input", dlex_cnn::CreateInputOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Output", dlex_cnn::CreateOutputOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("InnerProduct", dlex_cnn::CreateInnerProductOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Convolution", dlex_cnn::CreateConvolutionOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Deconvolution", dlex_cnn::CreateDeconvolutionOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Pooling", dlex_cnn::CreatePoolingOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Activation", dlex_cnn::CreateActivationOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("Softmax", dlex_cnn::CreateSoftmaxOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("CrossEntropyLoss", dlex_cnn::CreateCrossEntropyLossOp<float>);
	dlex_cnn::OpFactory<float>::getInstance().registerOpClass("SoftmaxCrossEntropyLossH", dlex_cnn::CreateSoftmaxCrossEntropyLossHOp<float>);
}

#endif  // DLEX_DLEX_HPP_
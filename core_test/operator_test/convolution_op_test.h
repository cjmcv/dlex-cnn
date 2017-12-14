////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test Convolution operator.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_CONV_HPP_
#define DLEX_TEST_CONV_HPP_

#ifdef USE_OP_TEST

#include "operator/convolution_op.h"
namespace dlex_cnn
{
	template <typename Dtype>
	class ConvolutionOpTest
	{
		template<typename T> friend class ConvolutionOp;

	public:
		ConvolutionOpTest() {};
		virtual ~ConvolutionOpTest() {};

	public:
		void exec();
	};
}
void testConv();

#endif //USE_OP_TEST

#endif

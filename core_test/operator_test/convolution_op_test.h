////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
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
		//template <float> friend class ConvolutionOp;//
		template<typename T> friend class ConvolutionOp;

	public:
		ConvolutionOpTest() {};
		virtual ~ConvolutionOpTest() {};

	public:
		void forward();
		void backward();
	};
}
void testConv();

#endif //USE_OP_TEST

#endif

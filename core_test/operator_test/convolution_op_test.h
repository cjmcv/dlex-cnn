////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_CONV_HPP_
#define DLEX_TEST_CONV_HPP_
//
#include "operator/convolution_op.h"

namespace dlex_cnn
{
	template <typename Dtype>
	class ConvolutionOpTest
	{
		//template <float> friend class ConvolutionOp;//
		template<typename Dtype> friend class ConvolutionOp;

	public:
		ConvolutionOpTest() {};
		virtual ~ConvolutionOpTest() {};

	public:
		void forward();
		void backward();
	};
}
	
void testConv();

#endif

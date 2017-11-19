////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_DECONV_HPP_
#define DLEX_TEST_DECONV_HPP_
//
#include "operator/deconvolution_op.h"

namespace dlex_cnn
{
	template <typename Dtype>
	class DeconvolutionOpTest
	{
		//template <float> friend class DeconvolutionOp;//
		template<typename Dtype> friend class DeconvolutionOp;

	public:
		DeconvolutionOpTest() {};
		virtual ~DeconvolutionOpTest() {};

	public:
		void forward();
		void backward();
	};
}
	
void testDeconv();

#endif

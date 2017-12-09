////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_DECONV_HPP_
#define DLEX_TEST_DECONV_HPP_

#ifdef USE_OP_TEST

#include "operator/deconvolution_op.h"
namespace dlex_cnn
{
	template <typename Dtype>
	class DeconvolutionOpTest
	{
		//template <float> friend class DeconvolutionOp;//
		template<typename T> friend class DeconvolutionOp;

	public:
		DeconvolutionOpTest() {};
		virtual ~DeconvolutionOpTest() {};

	public:
		void forward();
		void backward();
	};
}
void testDeconv();

#endif //USE_OP_TEST

#endif

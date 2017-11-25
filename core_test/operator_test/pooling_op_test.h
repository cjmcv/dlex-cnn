////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_POOLING_HPP_
#define DLEX_TEST_POOLING_HPP_
//
#include "operator/pooling_op.h"

namespace dlex_cnn
{
	template <typename Dtype>
	class PoolingOpTest
	{
		template<typename T> friend class PoolingOp;

	public:
		PoolingOpTest() {};
		virtual ~PoolingOpTest() {};

	public:
		void forward();
		void backward();
	};
}

void testPool();

#endif


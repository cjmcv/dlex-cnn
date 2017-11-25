////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_LOSS_HPP_
#define DLEX_OP_LOSS_HPP_

#include <vector>
#include <memory>
#include <stdlib.h>

#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct LossOpParam : public OpParam
	{

	};

	template <typename Dtype>
	class LossOp : public Op<Dtype>
	{
	public:
		explicit LossOp();
		virtual ~LossOp();

	};
}

#endif //DLEX_OP_HYBRID_HPP_
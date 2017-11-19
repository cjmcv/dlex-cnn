////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator_loss.h"

namespace dlex_cnn
{
	template <typename Dtype>
	LossOp<Dtype>::LossOp()
	{

	}
	template <typename Dtype>
	LossOp<Dtype>::~LossOp()
	{

	}
	INSTANTIATE_CLASS(LossOp);
}
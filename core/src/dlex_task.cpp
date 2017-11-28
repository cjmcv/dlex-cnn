////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_task.h"

namespace dlex_cnn
{
	template <typename Dtype>
	Task<Dtype>::Task()
	{
		
	}

	template <typename Dtype>
	Task<Dtype>::~Task()
	{

	}

	INSTANTIATE_CLASS(Task);
}
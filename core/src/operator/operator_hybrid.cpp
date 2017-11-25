////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/operator_hybrid.h"

const std::map<std::string, std::vector<std::string>, int> opListMapTwo;
const std::map<std::string, std::vector<std::string>, int> opListMapThree;


namespace dlex_cnn
{
	template <typename Dtype>
	HybridOp<Dtype>::HybridOp()
	{

	}
	template <typename Dtype>
	HybridOp<Dtype>::~HybridOp()
	{

	}
	template <typename Dtype>
	int HybridOp<Dtype>::setSubOp(const std::vector<std::shared_ptr<Op<Dtype>>> &sub_ops)
	{
		if (sub_ops.size() <= 0)
			return -1;
		sub_ops_.clear();
		for (int i = 0; i < sub_ops.size(); i++)
			sub_ops_.push_back(sub_ops[i]);
		return 0;
	}

	INSTANTIATE_CLASS(HybridOp);
}
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_HYBRID_HPP_
#define DLEX_OP_HYBRID_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>

#include "operator_base.h"
#include "tensor.h"
#include "util/op_factory.h"



namespace dlex_cnn
{

#ifndef HOP_PHASEMAP_NUM
#define HOP_PHASEMAP_NUM 2
#endif

#ifndef OP_DOUBLE_NUM
#define OP_DOUBLE_NUM 2
#endif

#ifndef OP_TRIPLET_NUM
#define OP_TRIPLET_NUM 3
#endif

	// MOVE those list to other place

	// dstOp, trainOp, testOp
	const std::string hopPhaseMap[][3] =
	{
		"SoftmaxCrossEntropyLossH", "SoftmaxCrossEntropyLossH", "Softmax",
		"SigmoidLoss", "SigmoidLoss", "Sigmoidss"
	};

	const std::string opListDouble[][4] = 
	{ 
		"SoftmaxCrossEntropyLossH", "Softmax", "CrossEntropyLoss", "0",
		"SigmoidLoss", "Sigmoidss", "CrossEntropyLoss", "1" 
	};

	const std::string opListTriplet[][5] = 
	{ 
		"asd", "fgh", "jkl", "jklsa", "0",
		"asd", "fgh", "jkl", "jklsa", "1",
		"asd", "fgh", "jkl", "jklsa", "2" 
	};

	struct HybridOpParam : public OpParam
	{

	};

	template <typename Dtype>
	class HybridOp : public Op<Dtype>
	{
	public:
		explicit HybridOp();
		virtual ~HybridOp();

		int setSubOp(const std::vector<std::shared_ptr<Op<Dtype>>> &sub_ops);

	protected:
		std::vector<std::shared_ptr<Op<Dtype>>> sub_ops_;
	};

}

#endif //DLEX_OP_HYBRID_HPP_
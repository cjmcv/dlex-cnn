////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_BASE_HPP_
#define DLEX_OP_BASE_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>

//#include "dlex_datatype.h"
#include "tensor.h"


namespace dlex_cnn
{
	namespace tind
	{
		enum OpCategory { eNormOp, eLossOp, eHybridOp };
	}

	struct OpParam
	{

	};

	template <typename Dtype>
	class Op
	{
	public:
		explicit Op();
		virtual ~Op();

		// Set operator's parameters by specific string from model file.
		virtual int setOpParam(const std::string &op_param_str) { return -1; };

		// Generate operator's parameter string for model saving.
		virtual std::string genOpParamStr() const { return ""; };

		virtual int inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) { return -1; };

		// Allocate memory buffer for node, including data/weight/blas.
		virtual int allocBuf4Node(const std::vector<int> &in_shape,
			const std::vector<int> &out_shape,
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const {
			return -1;
		};

		// Allocate memory buffer for training, including gradient and difference in operator
		virtual int allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) { return -1; };

		virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpGradient() { return gradient_; };
		virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() { return diff_; };
		virtual const std::string &getOpType() const { return " "; };
		virtual const int getOpCategory() const { return tind::eNormOp; };

		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {};
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {};
	
	private:
		// Gradient, including weightGradient/blasGradient, has the same size with weight.
		// Will be used in optimizer for updating weight/blas during backward operation.
		// Should be cleaned at each backward iteration
		std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;

		// Difference, has the same size with input data, will be used in backward operation for updating gradient;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		OpParam param_;
	};

}

#endif //DLEX_OP_BASE_HPP_
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_CROSS_ENTROPY_HPP_
#define DLEX_OP_CROSS_ENTROPY_HPP_

#include "configure.h"
#include "operator_base.h"
#include "operator_loss.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct CrossEntropyLossOpParam : public LossOpParam
	{

	};

	template <typename Dtype>
	class CrossEntropyLossOp : public LossOp<Dtype>
	{
	public:
		CrossEntropyLossOp();
		CrossEntropyLossOp(CrossEntropyLossOpParam param);
		virtual ~CrossEntropyLossOp();
		inline virtual int setOpParam(CrossEntropyLossOpParam opParam) { param_ = opParam; return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eLossOp; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpGradient() override { return gradient_; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() override { return diff_; };

		virtual std::string genOpParamStr() const override;
		virtual int inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape) override;
		//virtual int solveInnerParams(const std::vector<int> &inShape, const std::vector<int> &outShape,
		//	std::vector<std::shared_ptr<Tensor<Dtype>>> &data) override;
		virtual int allocBuf4Node(const std::vector<int> &inShape,
			const std::vector<int> &outShape, 
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
		virtual int allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape) override;

		// Different loss definition for different activation function: http://m.blog.csdn.net/u012436149/article/details/69660214
		//
		// prev[0] -> outdata of the last node (activation node).
		// next[1] -> orgLabelData, has been fetched from label file, will be converted to fit this loss operation.
		// next[2] -> ouput loss.
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
		
		// prevDiff[0] -> gradients for backward.
		// next[0] -> outdata of the last node.
		// prev and nextDiff is inactive in this function.
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff) override;

	private:
		std::string op_type_;

		std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;

		CrossEntropyLossOpParam param_;

		// Have the matched dimensions to this loss operator
		// Will be converted by the labels data in outNode
		std::shared_ptr<Tensor<Dtype>> labels_;
	};

	/////////////////////////////  Creator   /////////////////////////////////
	template <typename Dtype>
	CrossEntropyLossOp<Dtype> *CreateOp(CrossEntropyLossOpParam &param)
	{
		CrossEntropyLossOp<Dtype> *op = NULL;
		op = new CrossEntropyLossOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateCrossEntropyLossOp()
	{
		return std::shared_ptr<Op<Dtype>>(new CrossEntropyLossOp<Dtype>());
	}
}
#endif
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_SOFTMAX_HPP_
#define DLEX_OP_SOFTMAX_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct SoftmaxOpParam : public OpParam
	{

	};

	template <typename Dtype>
	class SoftmaxOp : public Op<Dtype>
	{
	public:
		SoftmaxOp();
		SoftmaxOp(SoftmaxOpParam param);
		virtual ~SoftmaxOp();	
		inline virtual int setOpParam(SoftmaxOpParam opParam) { param_ = opParam; return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eNormOp; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() override { return diff_; };

		virtual std::string genOpParamStr() const override;
		virtual int inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape) override;
		virtual int allocBuf4Node(const std::vector<int> &inShape,
			const std::vector<int> &outShape,
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
		virtual int allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape) override;
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff) override;

	private:
		std::string op_type_;
		SoftmaxOpParam param_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		
	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	SoftmaxOp<Dtype> *CreateOp(SoftmaxOpParam &param)
	{
		SoftmaxOp<Dtype> *op = NULL;
		op = new SoftmaxOp<Dtype>(param);
		return op;
	}

	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateSoftmaxOp()
	{
		return std::shared_ptr<Op<Dtype>>(new SoftmaxOp<Dtype>());
	}
}
#endif
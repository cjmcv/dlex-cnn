////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_OUTPUT_HPP_
#define DLEX_OP_OUTPUT_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct OutputOpParam : public OpParam
	{
		int label_dim;
	};

	template <typename Dtype>
	class OutputOp : public Op<Dtype>
	{
		//FRIEND_WITH_NETWORK
	public:
		OutputOp();
		OutputOp(OutputOpParam param);
		virtual ~OutputOp();
		inline virtual int setOpParam(OutputOpParam opParam) { param_ = opParam; return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		//DECLARE_LAYER_TYPE;
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
		OutputOpParam param_;
		//std::vector<std::shared_ptr<Tensor<float>>> gradient_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	OutputOp<Dtype> *CreateOp(OutputOpParam &param)
	{
		OutputOp<Dtype> *op = NULL;
		op = new OutputOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateOutputOp()
	{
		return std::shared_ptr<Op<Dtype>>(new OutputOp<Dtype>());
	}
}
#endif
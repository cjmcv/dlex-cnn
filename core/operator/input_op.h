////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_INPUT_HPP_
#define DLEX_OP_INPUT_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct InputOpParam : public OpParam
	{
		InputOpParam() {};
		InputOpParam(int in_num, int in_channels, int in_height, int in_width)
		{
			num = in_num;
			channels = in_channels;
			height = in_height;
			width = in_width;
		}
		int num;
		int channels;
		int height;
		int width;
	};

	template <typename Dtype>
	class InputOp : public Op<Dtype>
	{
		//FRIEND_WITH_NETWORK
	public:
		InputOp();
		InputOp(InputOpParam param);
		virtual ~InputOp();
		inline virtual int setOpParam(InputOpParam opParam) { param_ = opParam; return 0; };
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
		InputOpParam param_;
		//std::vector<std::shared_ptr<Tensor<float>>> gradient_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	InputOp<Dtype> *CreateOp(InputOpParam &param)
	{
		InputOp<Dtype> *op = NULL;
		op = new InputOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateInputOp()
	{
		return std::shared_ptr<Op<Dtype>>(new InputOp<Dtype>());
	}
}
#endif
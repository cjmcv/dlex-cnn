////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_SOFTMAX_HPP_
#define DLEX_OP_SOFTMAX_HPP_

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
		inline virtual int setOpParam(SoftmaxOpParam op_param) { param_ = op_param; return 0; };
		virtual int setOpParam(const std::string &op_param_str) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eNormOp; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() override { return diff_; };

		virtual std::string genOpParamStr() const override;
		virtual int inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) override;
		virtual int allocBuf4Node(const std::vector<int> &in_shape,
			const std::vector<int> &out_shape,
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
		virtual int allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) override;

		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) override;
#ifdef USE_CUDA
		virtual void forward_gpu(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
		virtual void backward_gpu(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) override;
#endif

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

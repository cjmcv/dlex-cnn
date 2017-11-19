////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_ACTIVATION_HPP_
#define DLEX_OP_ACTIVATION_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"
#include "util/math_functions.h"
#include <functional> 
#include <algorithm>

namespace dlex_cnn
{
	namespace tind
	{
		enum Activation { eReLU, eSigmoid, eTanh };
	}

	struct ActivationOpParam : public OpParam
	{
		tind::Activation activationType = tind::eReLU;

		// relu
		float negative_slope = 0;

		// prelu
		//
	};

	template <typename Dtype>
	class ActivationOp : public Op<Dtype>
	{
		//FRIEND_WITH_NETWORK
	public:
		ActivationOp();
		ActivationOp(ActivationOpParam param);
		virtual ~ActivationOp();
		inline virtual int setOpParam(ActivationOpParam opParam) { param_ = opParam; setOpFunc(); return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		std::function<Dtype(Dtype)> pAct;
		std::function<Dtype(Dtype, Dtype)> pRevAct;

		inline Dtype relu(Dtype x) { return std::max(x, Dtype(0)) + param_.negative_slope * std::min(x, Dtype(0)); }
		inline Dtype rev_relu(Dtype px, Dtype diff_next) { return diff_next * ((px > 0) + param_.negative_slope * (px <= 0)); }

		inline Dtype sigmoid(Dtype x) { return 1. / (1. + exp(-x)); }
		inline Dtype rev_sigmoid(Dtype nx, Dtype diff_next) { return diff_next * nx * (1. - nx); }

		inline Dtype tanh(Dtype x) { return std::tanh(x); }
		inline Dtype rev_tanh(Dtype nx, Dtype diff_next) { return diff_next * (1 - nx * nx); }

	private:
		int setOpFunc();

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
		ActivationOpParam param_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
	};


	/////////////////////////////  Creator   /////////////////////////////////
	template <typename Dtype>
	ActivationOp<Dtype> *CreateOp(ActivationOpParam &param)
	{
		ActivationOp<Dtype> *op = NULL;
		op = new ActivationOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateActivationOp()
	{
		return std::shared_ptr<Op<Dtype>>(new ActivationOp<Dtype>());
	}
}
#endif
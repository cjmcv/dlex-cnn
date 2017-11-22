////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_SOFTMAX_CROSS_HPP_
#define DLEX_OP_SOFTMAX_CROSS_HPP_

#include "configure.h"
#include "operator_base.h"
#include "operator_hybrid.h"
#include "operator/cross_entropy_lop.h"
#include "operator/softmax_op.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct SoftmaxCrossEntropyLossHOpParam : public HybridOpParam
	{

	};

	//////////////////////////////////////////////////////////////////////////
	// cross entropy
	// -- 加载网络结构时，为每个node建立索引，可以直接索引到loss的node，每个loss node的后面都接自动一个outputNode作为缓存。
	// outputNode带Op，无前后向传播，只带data和diff；data[0]为output/[1]为label/[2]为loss，diff分配空间用于反向。
	// 定义一个node，加softmaxLoss，由softmaxLoss附加softmaxOp，以softmaxLoss来softmaxOp。
	// 前向node的cpu_data为数据，作为prev，输出为softmax的output与loss，下一个node（operator_output node）。
	// 反向获得diff：prev、next(output+label，在outNode的data中)、prevdiff、nextdiff(outNode的diff)
	// 前向反向的操作子均在prev

	// 前向（总），输入上一层数据，输出output[0],label[1]和loss[2]
	// 前向，softmax输入上一层数据，输出output。loss输入output，从输出的[1]中取label,输出loss；转存两个结果

	// 反向（总），prev、next(output+label，在outNode的data中)、prevdiff、nextdiff(outNode的diff，只传不取)
	// 反向，getDiff： prev、next(output+label，在outNode的data中)、prevdiff(放nextdiff)、nextdiff(outNode的diff，只传不取)， 将得到的diff置prevdiff
	// 反向，softmax： prev, lastOutput_（outNode的data，内取0）, prevdiff, lastDiff_（getDiff的结果）

	template <typename Dtype>
	class SoftmaxCrossEntropyLossHOp : public HybridOp<Dtype>
	{
	public:
		SoftmaxCrossEntropyLossHOp();
		SoftmaxCrossEntropyLossHOp(SoftmaxCrossEntropyLossHOpParam param);
		virtual ~SoftmaxCrossEntropyLossHOp();
		inline virtual int setOpParam(SoftmaxCrossEntropyLossHOpParam op_param) { param_ = op_param; return 0; };
		virtual int setOpParam(const std::string &op_param_str) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eHybridOp; };
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

	private:
		std::string op_type_;
		SoftmaxCrossEntropyLossHOpParam param_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		//std::vector<std::shared_ptr<Op<Dtype>>> sub_ops_;
	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	SoftmaxCrossEntropyLossHOp<Dtype> *CreateOp(SoftmaxCrossEntropyLossHOpParam &param)
	{
		SoftmaxCrossEntropyLossHOp<Dtype> *op = NULL;
		op = new SoftmaxCrossEntropyLossHOp<Dtype>(param);
		return op;
	}

	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateSoftmaxCrossEntropyLossHOp()
	{
		return std::shared_ptr<Op<Dtype>>(new SoftmaxCrossEntropyLossHOp<Dtype>());
	}
}
#endif
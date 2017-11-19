////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_POOLING_HPP_
#define DLEX_OP_POOLING_HPP_

#include "configure.h"
#include "operator_base.h"
#include "util/math_functions.h"
#include "tensor.h"

#ifdef UNIT_TEST
#include "../core_test/operator_test/pooling_op_test.h"
#endif


namespace dlex_cnn
{
	namespace tind
	{
		enum PoolingType { eMAX, eAVE, eSTOCHASTIC };
	}
	struct PoolingOpParam : public OpParam
	{
		tind::PoolingType poolingType = tind::eMAX;
		int kernel_h = 3, kernel_w = 3;
		int stride_h = 1, stride_w = 1;
		int pad_h = 0, pad_w = 0;
		//int channels;
		//int height_, width_;	//输入图像数据大小
		//int pooled_height_, pooled_width_;	//实际池化输出大小，static_cast<int>(ceil(static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
		bool global_pooling = false;	//为true时，kernel大小为输入大小
	};

	template <typename Dtype>
	class PoolingOp : public Op<Dtype>
	{
#ifdef UNIT_TEST
		template <typename Dtype>
		friend class PoolingOpTest;
#endif
	public:
		PoolingOp();
		PoolingOp(PoolingOpParam param);
		virtual ~PoolingOp();
		inline virtual int setOpParam(PoolingOpParam opParam) { param_ = opParam; return 0; };
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

		// Refer to Caffe
		// In max pooling, it should mark the location(index) of max number in each sliding window.
		// In average pooling, just get the mean value;
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
		// In max pooling, According to the mask we have got in forward, we just only update the points which is the max of each sliding windows in forward.
		// In average pooling, each next value will be divided by kernel size, and fill the relevant sliding window with the quotient;
		// If sliding windows are overlap, the output will be accumulated.
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff) override;

	private:
		std::string op_type_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		std::shared_ptr<Tensor<Dtype>> rand_idx_map_;
		std::shared_ptr<Tensor<int>> max_idx_map_;
		PoolingOpParam param_;

	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	PoolingOp<Dtype> *CreateOp(PoolingOpParam &param)
	{
		PoolingOp<Dtype> *op = NULL;
		op = new PoolingOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreatePoolingOp()
	{
		return std::shared_ptr<Op<Dtype>>(new PoolingOp<Dtype>());
	}
}
#endif
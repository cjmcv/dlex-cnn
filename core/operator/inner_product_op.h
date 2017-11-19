////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_INNER_PRODUCT_HPP_
#define DLEX_OP_INNER_PRODUCT_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn
{
	struct InnerProductOpParam : public OpParam
	{
		bool blas_enable;
		int num_hidden;
	};

	template <typename Dtype>
	class InnerProductOp : public Op<Dtype>
	{
	public:
		InnerProductOp();
		InnerProductOp(InnerProductOpParam param);
		virtual ~InnerProductOp();
		inline virtual int setOpParam(InnerProductOpParam opParam) { param_ = opParam; return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eNormOp; };
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
		
		// fc: o1 = i1 x w11 + i2 x w12 + i3 x w13 + b1;
		//     o2 = i1 x w21 + i2 x w22 + i3 x w23 + b2;
		//     o3 = i1 x w31 + i2 x w32 + i3 x w33 + b3;
		// ->  matrix O = I * W + Blas  -> can use GEMM to implement
		//
		// input(num, channels, height, width) as matrix I(M,K) = I(num, channels*height*width),
		// weight as matrix W(N, K) = W(hidden_num, channels*height*width),
		// blas as matrix B(N) = B(hidden_num),
		// output(num, hidden_num, 1, 1) as matrix O(M,N) = O(num, hidden_num),
		// ->  O = I * W' + B. (Our case).
		//
		// note: In caffe, there has a variable "transpose_". 
		//		 If the variable is "true", assume transposed weights, that means weights have been saved like W(K,N). The formula should be O = I * W + B.
		//       Or if it is "false", then weights have been saved like W(K,N), and the formula will be changed to O = I * W' + B.
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;

		// 1��Update prevDiff, according to nextDiff and weight.
		// prevDiff(num, in3DSize) = nextDiff(num, hidden_num) * weight(hidden_num, in3DSize).
		// -> prevDiff(num, prevDiffSize[tind::e3D]) = nextDiff(num, nextDiffSize[tind::e3D]) * weight(nextDiffSize[tind::e3D], in3DSize).
		//
		// 2��Get weight gradient.
		// nextDiff(num, hidden_num) -> nextDiff'(hidden_num, num).
		// O(M,N) = weightGradient(hidden_num, in3DSize) = nextDiff'(hidden_num, num) * prevData(num, in3DSize).
		// -> M=hidden_num, N=in3DSize, K=num.
		//
		// 3��update bias, just accumulate nextDiffData.
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff) override;

		//void setParamaters(const DataSize _outMapSize, const bool _enabledBias);

	private:
		std::string op_type_;

		std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		InnerProductOpParam param_;
		
	};

	/////////////////////////////  Creator   /////////////////////////////////

	template <typename Dtype>
	InnerProductOp<Dtype> *CreateOp(InnerProductOpParam &param)
	{
		InnerProductOp<Dtype> *op = NULL;
		op = new InnerProductOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateInnerProductOp()
	{
		return std::shared_ptr<Op<Dtype>>(new InnerProductOp<Dtype>());
	}
}
#endif
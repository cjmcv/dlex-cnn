////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_DECONVOLUTION_HPP_
#define DLEX_OP_DECONVOLUTION_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"
#include "util/math_functions.h"

#ifdef UNIT_TEST
#include "../core_test/operator_test/deconvolution_op_test.h"
#endif

namespace dlex_cnn
{
	struct DeconvolutionOpParam : public OpParam
	{
		bool blas_enable = true;
		//int kernel_num = 1;	// kernel_num is equal to input channels.
		int kernel_channels = 1;
		int kernel_h = 3, kernel_w = 3;
		int stride_h = 1, stride_w = 1;
		int pad_h = 0, pad_w = 0;
		int dilation_h = 1, dilation_w = 1;
	};

	template <typename Dtype>
	class DeconvolutionOp : public Op<Dtype>
	{
#ifdef UNIT_TEST
		template <typename Dtype>
		friend class DeconvolutionOpTest;
#endif
	public:
		DeconvolutionOp();
		DeconvolutionOp(DeconvolutionOpParam param);
		virtual ~DeconvolutionOp();
		inline virtual int setOpParam(DeconvolutionOpParam opParam) { param_ = opParam; return 0; };
		virtual int setOpParam(const std::string &opParamStr) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eNormOp; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpGradient() override { return gradient_; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() override { return diff_; };

		virtual std::string genOpParamStr() const override;
		virtual int inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape) override;
		virtual int allocBuf4Node(const std::vector<int> &inShape,
			const std::vector<int> &outShape,
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
		virtual int allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape) override;

		// conv (prev channels == kernel channels, kernel num == next channels)
		// deconv (prev channels == kernel num, kernel channels == next channels). Just reverse the place of prev and next in conv
		//
		// conv (n->1); deconv (1->n)
		// input: 1,0,0,0  (dconv)  kernel: 4,5  ==>  output: 4, 5, 0, 0, 0
	    //        0,1,0,0                   3,4               3, 8, 5, 0, 0
		//		  0,0,3,0                                     0, 3,16,15, 0
		//		  0,0,0,1                                     0, 0, 9,16, 5
		//                                                    0, 0, 0, 3, 4
		//
		// kernel'*prev = col, col2im = next
		// col(kc*kh*kw, ph*pw) = kernel'(kc*kh*kw, kn=pc) * prev(pc, ph*pw)
		// col2im(nc=kc, nh*nw) = col(kc*kh*kw, ph*pw)
		// ps: nh/nw, refer to inferOutShape.
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;

		// 1, update prevDiff
		//    col2im(nc=kc, nh*nw) = col(kc*kh*kw, ph*pw)
		//    prevDiff(pc=kn, ph*pw) = kernel(kn, kc*kh*kw) * nextDiff_col(kc*kh*kw, ph*pw)
		//
		// 2, update weight Diff
		//    kernelGradient(kn=pc, kc*kh*kw) = prev(pc, ph*pw) * nextDiff_col'(ph*pw, kc*kh*kw);
		//    ps: in conv -> kernelGradient = nextDiff * bottom_col'
		//
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff) override;

	private:
		std::string op_type_;

		std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;	// kernel/bias
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		std::shared_ptr<Tensor<Dtype>> col_buffer_;

		DeconvolutionOpParam param_;
	};

	/////////////////////////////   Creator   /////////////////////////////////
	template <typename Dtype>
	DeconvolutionOp<Dtype> *CreateOp(DeconvolutionOpParam &param)
	{
		DeconvolutionOp<Dtype> *op = NULL;
		op = new DeconvolutionOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateDeconvolutionOp()
	{
		return std::shared_ptr<Op<Dtype>>(new DeconvolutionOp<Dtype>());
	}
}
#endif
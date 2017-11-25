////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_CONVOLUTION_HPP_
#define DLEX_OP_CONVOLUTION_HPP_

#include "configure.h"
#include "operator_base.h"
#include "tensor.h"
#include "util/math_functions.h"

#ifdef UNIT_TEST
#include "../../core_test/operator_test/convolution_op_test.h"
#endif

namespace dlex_cnn
{
	struct ConvolutionOpParam : public OpParam
	{
		bool blas_enable = true;
		int kernel_num = 1;
		//int kernel_channels = 1;	// kernel_channels is equal to input channels.
		int kernel_h = 3, kernel_w = 3;
		int stride_h = 1, stride_w = 1;
		int pad_h = 0, pad_w = 0;
		int dilation_h = 1, dilation_w = 1;	
		// ps: VALID - round down, if data's demision can not be divide by kernel size exactly, delete the rest; SAME - round up
	};

	template <typename Dtype>
	class ConvolutionOp : public Op<Dtype>
	{
#ifdef UNIT_TEST
		template <typename T>
		friend class ConvolutionOpTest; // the typename should not be the same width ConvoluitionOp's "Dtype"
#endif
	public:
		ConvolutionOp();
		ConvolutionOp(ConvolutionOpParam param);
		virtual ~ConvolutionOp();
		inline virtual int setOpParam(ConvolutionOpParam op_param) { param_ = op_param; return 0; };
		virtual int setOpParam(const std::string &op_param_str) override;

	private:
		inline virtual const std::string &getOpType() const override { return op_type_; };
		inline virtual const int getOpCategory() const override { return tind::eNormOp; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpGradient() override { return gradient_; };
		inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &getOpDiff() override { return diff_; };

		virtual std::string genOpParamStr() const override;
		virtual int inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) override;
		virtual int allocBuf4Node(const std::vector<int> &in_shape,
			const std::vector<int> &out_shape,
			std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
		virtual int allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) override;

		// Refer to the paper "High Performance Convolutional Neural Networks for Document Processing"
		///////// origin convolution /////////
		// input(pc3, ph3, pw3):               1 2 0  |  0 2 1  |  1 2 1        ->    in memory: 1 2 0 1 1 3 0 2 2 , 0 2 1 0 3 2 1 1 0 ...
		//		                               1 1 3  |  0 3 2  |  0 1 3
		//                                     0 2 2  |  1 1 0  |  3 3 2
		//
		// kernel(kn2, kc3 = pc3, kh2, kw2):   1 1  |  1 1  |  0 1   -   1 0  |  2 1  |  1 2     ->    in memory: 1 1 2 2 , 1 1 1 1 , 0 1 1 0 ,, 1 0 0 1 ...
		//		                               2 2  |  1 1  |  1 0   -   0 1  |  2 1  |  2 0
		//
		//
		// output(nc2 = kn2, nh2, nw2):        14 20  |  12 24     ->    in memory:  14 20 15 24 , 12 24 17 26 ...
		//                                     15 24  |  17 26
		// ps: 14 = 1*1+2*1+1*2+1*2 + 0*1+2*1+0*1+3*1 + 1*0+2*1+0*1+1*0     |    12 = 1*1+2*0+1*0+1*1 + 0*2+2*1+0*2+3*1 + 1*1+2*2+0*2+1*0
		//
		///////// implement /////////
		// input(pc3, ph3, pw3):                              1 2 1 1 , 0 2 0 3 , 1 2 0 1  (transpose)  ->      ps: 1 2 -> 1 2 1 1 (T) |  0 2 -> 0 2 0 3 (T)
		//    ->(12 , 4)                                      2 0 1 3 , 2 1 3 2 , 2 1 1 3                           1 1                   0 3
		//    -> 4 = r(ph3-(kh2+1)/2) * r(pw3-(kw2+1)/2)      1 1 0 2 , 0 3 1 1 , 0 1 3 3                           2 0 -> 2 0 1 3 (T) |  1 3 -> 1 3 3 2 (T)
		//    ->12 = kh2*kw2*pc3                              1 3 2 2 , 3 2 1 0 , 1 3 3 2                           1 3                   3 2
		//
		// kernel(kn2, kc3 = pc3, kh2, kw2):     1 1 2 2 , 1 1 1 1 , 0 1 1 0    ->  in memory (needn't to be transposed): 1 1 2 2 , 1 1 1 1 , 0 1 1 0 ,, 1 0 0 1 ...
		//     ->(2=kn2 , 12 = kh2*kw2*ic3)      1 0 0 1 , 2 1 2 1 , 1 2 2 0
		//
		// output(2,4)->kernel(2,12)*input(12,4)     14 20 15 24   ->  in memory (needn't to be transposed):  14 20 15 24 , 12 24 17 26 ...
		//                                           12 24 17 26
		virtual void forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;

		// 1, update prev_diff
		//    prevDiff_img() = prevDiff_col(kc*kh*kw, nh*nw) = kernel'(kc*kh*kw, kn) * next_diff(nc, nh*nw)
		//                     -> the num of kernels is equal to the channels of output. (ps: pc == kc, kn == nc)
		//    As to im2col, col_h = (img_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1, and col_w is similar.
		//
		// 2, update weight Diff
		//    weightGradient(kn, kc*kh*kw) = next_diff(nc=kn, nh*nw) * prevData_col'(nh*nw, kc*kh*kw)
		//
		virtual void backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
			const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) override;

	private:
		std::string op_type_;

		std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;	// kernel/bias
		std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
		std::shared_ptr<Tensor<Dtype>> col_buffer_;

		ConvolutionOpParam param_;
	};

	/////////////////////////////   Creator   /////////////////////////////////
	template <typename Dtype>
	ConvolutionOp<Dtype> *CreateOp(ConvolutionOpParam &param)
	{
		ConvolutionOp<Dtype> *op = NULL;
		op = new ConvolutionOp<Dtype>(param);
		return op;
	}
	template <typename Dtype>
	std::shared_ptr<Op<Dtype>> CreateConvolutionOp()
	{
		return std::shared_ptr<Op<Dtype>>(new ConvolutionOp<Dtype>());
	}
}
#endif

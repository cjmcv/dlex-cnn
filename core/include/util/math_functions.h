////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_MATH_HPP_
#define DLEX_OP_MATH_HPP_

#include <string.h>
#include <cmath>
#include <random>
#include <algorithm>

namespace dlex_cnn
{
	//all of these functions below is run on single thread, maybe they are SIMD optimized.
	template <typename Dtype>
	void normal_distribution_init(Dtype* data, const int size, const Dtype mean_value, const Dtype standard_deviation);

	template <typename Dtype>
	void dlex_set(Dtype* data, const int N, const Dtype alpha);

	//a /= b
	template <typename Dtype>
	void div_inplace(Dtype* a, const Dtype b, const int len);

	template <typename Dtype>
	void im2col_cpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_col);

	template <typename Dtype>
	void col2im_cpu(const Dtype* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_im);

	template <typename Dtype>
	void gemm(bool bTransA, bool bTransB, const int M, const int N, const int K, const float alpha, const Dtype* A, const Dtype* B, const float beta, Dtype* C);

	template <typename Dtype>
	void add_bias(const int num, const int len, const Dtype* bias, Dtype* dst);
	template <typename Dtype>
	void add_bias(const int num, const int ch_size, const int len, const Dtype* bias, Dtype* dst);

	template <typename Dtype>
	void backward_bias(const int num, const int len, const Dtype* next_diff, Dtype* bias_gradient);
	template <typename Dtype>
	void backward_bias(const int num, const int ch_size, const int len, const Dtype* next_diff, Dtype* bias_gradient);
};
#endif

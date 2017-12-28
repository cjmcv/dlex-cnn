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
#include "util/device.h"

namespace dlex_cnn
{
	template <typename Dtype>
	void normal_distribution_init(const int size, const Dtype mean_value, const Dtype standard_deviation, Dtype* data);

	template <typename Dtype>
	void set_cpu(const int N, const Dtype alpha, Dtype* data);

	//a /= b
	template <typename Dtype>
	void div_inplace_cpu(const int N, const Dtype alpha, Dtype* data);

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
	void gemm_cpu(bool bTransA, bool bTransB, const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);

	template <typename Dtype>
	void add_bias(const int num, const int len, const Dtype* bias, Dtype* dst);
	template <typename Dtype>
	void add_bias(const int num, const int ch_size, const int len, const Dtype* bias, Dtype* dst);

	template <typename Dtype>
	void backward_bias(const int num, const int len, const Dtype* next_diff, Dtype* bias_gradient);
	template <typename Dtype>
	void backward_bias(const int num, const int ch_size, const int len, const Dtype* next_diff, Dtype* bias_gradient);

#ifdef USE_CUDA
	template <typename Dtype>
	void set_gpu(const int N, const Dtype alpha, Dtype* data);

	template <typename Dtype>
	void div_inplace_gpu(const int N, const Dtype alpha, Dtype* data);

	template <typename Dtype>
	void dlex_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma, Dtype* r);

	template <typename Dtype>
	void gemm_gpu(cublasHandle_t cublas_handle, const bool TransA,
		const bool TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
		Dtype* C);

	template <typename Dtype>
	void add_bias_gpu(const int num, const int len, const Dtype* bias, Dtype* dst);
	template <typename Dtype>
	void add_bias_gpu(const int num, const int ch_size, const int len, const Dtype* bias, Dtype* dst);

	template <typename Dtype>
	void backward_bias_gpu(const int num, const int len, Dtype* next_diff, Dtype* bias_gradient);
	template <typename Dtype>
	void backward_bias_gpu(const int num, const int ch_size, const int len, Dtype* next_diff, Dtype* bias_gradient);

	template <typename Dtype>
	void im2col_gpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w, Dtype* data_col);
	template <typename Dtype>
	void col2im_gpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w, Dtype* data_col);
#endif
};
#endif

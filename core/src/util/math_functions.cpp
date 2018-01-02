////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  math_function.cpp
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/math_functions.h"

namespace dlex_cnn
{
	template <typename Dtype>
	void normal_distribution_init(const int size, const Dtype mean_value, const Dtype standard_deviation, Dtype* data)
	{
		std::random_device rd;
		std::mt19937 engine(rd());
		std::normal_distribution<Dtype> dist(mean_value, standard_deviation);	//uniform_real_distribution
		for (int i = 0; i < size; i++)
		{
			data[i] = dist(engine);
		}
	}
	template void normal_distribution_init<float>(const int size, const float mean_value, const float standard_deviation, float* data);
	template void normal_distribution_init<double>(const int size, const double mean_value, const double standard_deviation, double* data);

	template <typename Dtype>
	void set_cpu(const int N, const Dtype alpha, Dtype* data)
	{
		if (alpha == 0) 
		{
			memset(data, 0, sizeof(Dtype) * N);
			return;
		}
		for (int i = 0; i < N; ++i) 
			data[i] = alpha;
	}
	template void set_cpu<int>(const int N, const int alpha, int* data);
	template void set_cpu<float>(const int N, const float alpha, float* data);
	template void set_cpu<double>(const int N, const double alpha, double* data);

	//a /= b
	template <typename Dtype>
	void div_inplace_cpu(const int N, const Dtype alpha, Dtype* data)
	{
		for (int i = 0; i < N; i++)
		{
			data[i] /= alpha;
		}
	}
	template void div_inplace_cpu<int>(const int N, const int alpha, int* data);
	template void div_inplace_cpu<float>(const int N, const float alpha, float* data);
	template void div_inplace_cpu<double>(const int N, const double alpha, double* data);


	// A(M,K) * B(K, N) = C(M, N)
	template <typename Dtype>
	void gemm_cpu(bool bTransA, bool bTransB, const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C)
	{
		int i, j, t;
		int lda = (bTransA == false) ? K : M;
		int ldb = (bTransB == false) ? N : K;
		
		for (i = 0; i < M; i++)
			for (j = 0; j < N; j++)
				C[i*N + j] *= beta;

		if (bTransA == false && bTransB == true)
		{
			for (i = 0; i < M; i++)
			{
				for (j = 0; j < N; j++)
				{
					Dtype val = 0;
					for (t = 0; t < K; t++)
					{
						val += A[i*lda + t] * B[j*ldb + t];
					}
					C[i*N + j] += val;
				}
			}
		}
		else if (bTransA == false && bTransB == false)
		{
			for (i = 0; i < M; i++)
			{
				for (j = 0; j < N; j++)
				{
					Dtype val = 0;
					for (t = 0; t < K; t++)
					{
						val += A[i*lda + t] * B[t*ldb + j];
					}
					C[i*N + j] += val;
				}
			}
		}
		else if (bTransA == true && bTransB == false)	//相当于 将A转置后与B相乘
		{
			for (i = 0; i < M; i++)
			{
				for (j = 0; j < N; j++)
				{
					Dtype val = 0;
					for (t = 0; t < K; t++)
					{
						val += A[t*lda + i] * B[t*ldb + j];
					}
					C[i*N + j] += val;
				}
			}
		}
	}
	template void gemm_cpu<float>(bool bTransA, bool bTransB, const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C);
	template void gemm_cpu<double>(bool bTransA, bool bTransB, const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C);

	template <typename Dtype>
	void add_bias(const int num, const int len, const Dtype* bias, Dtype* dst)
	{
		int i, j;
		for (i = 0; i < num; i++)
		{
			for (j = 0; j < len; j++)
			{
				dst[i*len + j] += bias[j];
			}
		}
	}
	template void add_bias<float>(const int num, const int len, const float* bias, float* dst);
	template void add_bias<double>(const int num, const int len, const double* bias, double* dst);

	template <typename Dtype>
	void add_bias(const int num, const int ch_size, const int len, const Dtype* bias, Dtype* dst)
	{
		int i, j, k;
		for (i = 0; i < num; i++)
		{
			for (j = 0; j < ch_size; j++)
			{
				Dtype b = bias[j];
				for (k = 0; k < len; k++)
				{
					dst[i*ch_size*len + j*len + k] += b;
				}
			}
		}
	}
	template void add_bias<float>(const int num, const int ch_size, const int len, const float* bias, float* dst);
	template void add_bias<double>(const int num, const int ch_size, const int len, const double* bias, double* dst);

	template <typename Dtype>
	void backward_bias(const int num, const int len, const Dtype* next_diff, Dtype* bias_gradient)
	{
		int n, i;
		for (n = 0; n < num; n++)
		{
			for (i = 0; i < len; i++)
			{
				bias_gradient[i] += 1.0f*next_diff[n * len + i]; //1.0f*next_diff_data[n * len + i];
			}
		}
	}
	template void backward_bias<float>(const int num, const int len, const float* next_diff, float* bias_gradient);
	template void backward_bias<double>(const int num, const int len, const double* next_diff, double* bias_gradient);

	template <typename Dtype>
	void backward_bias(const int num, const int ch_size, const int len, const Dtype* next_diff, Dtype* bias_gradient)
	{
		int n,c,k;
		for (n = 0; n < num; n++)
		{
			for (c = 0; c < ch_size; c++)
			{
				Dtype b = 0;
				for (k = 0; k < len; k++)
				{
					b += 1.0f*next_diff[n*ch_size*len + c*len + k];
				}
				bias_gradient[c] += b;
			}
		}
	}
	template void backward_bias<float>(const int num, const int ch_size, const int len, const float* next_diff, float* bias_gradient);
	template void backward_bias<double>(const int num, const int ch_size, const int len, const double* next_diff, double* bias_gradient);

	// Refer to Caffe
	// Function uses casting from int to unsigned to compare if value of
	// parameter a is greater or equal to zero and lower than value of
	// parameter b. The b parameter is of type signed and is always positive,
	// therefore its value is always lower than 0x800... where casting
	// negative value of a parameter converts it to value higher than 0x800...
	// The casting allows to use one condition instead of two.
	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
		return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}

	// Refer to Caffe
	// data_im: input matrix (channels, height, width)
	// data_col: output matrix (1, channels*kernel_h*kernel_w, output_h*output_w)
	//                      -> *(data_col++) will be executed (channels*kernel_h*kernel_w*output_h*output_w) times
	template <typename Dtype>
	void im2col_cpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_col) 
	{
		const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;	// The size of conv output, also means the sliding windows' number in vertical direction. 
		const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;	// In horizontal direction
		const int channel_size = height * width;
		for (int channel = channels; channel--; data_im += channel_size)
		{
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) 
			{
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) 
				{
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--)
					{
						if (!is_a_ge_zero_and_a_lt_b(input_row, height))	// If (input_row >= height || input_row < 0), that means it is pad and should be setted to zero.
						{
							for (int output_cols = output_w; output_cols; output_cols--)
								*(data_col++) = 0;
						}
						else
						{
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; output_col--)
							{
								if (is_a_ge_zero_and_a_lt_b(input_col, width))	  //If (input_col >= 0 && input_col < width), that means it is not pad.
									*(data_col++) = data_im[input_row * width + input_col];		//copy the real data from input.
								else 
									*(data_col++) = 0;		// Else it should be setted to zero.
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}

	// Explicit instantiation
	template void im2col_cpu<float>(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		float* data_col);
	template void im2col_cpu<double>(const double* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		double* data_col);

	//////////////////////////////////////////////////
	// 在deconv中，该函数用在forward过程，output为以实际输出进行卷积后应得到的维度，与data_col对应，即由该维度的矩阵可直接化为该col；入参channels/height/width为实际输出维度，与data_im对应
	template <typename Dtype>
	void col2im_cpu(const Dtype* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_im) {
		memset(data_im, 0, sizeof(Dtype) * height * width * channels);
		const int output_h = (height + 2 * pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int channel_size = height * width;
		for (int channel = channels; channel--; data_im += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							data_col += output_w;
						}
						else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									data_im[input_row * width + input_col] += *data_col;
								}
								data_col++;
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}

	// Explicit instantiation
	template void col2im_cpu<float>(const float* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		float* data_im);
	template void col2im_cpu<double>(const double* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		double* data_im);

}//namespace
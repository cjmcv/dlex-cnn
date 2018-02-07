////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  math_function.cu
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifdef USE_CUDA
#include "util/device.h"
#include "util/math_functions.h"

namespace dlex_cnn {
template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data[index] = alpha;
  }
}

template <typename Dtype>
void set_gpu(const int N, const Dtype alpha, Dtype* data) {
  if (alpha == 0) {
    CUDA_DCHECK(cudaMemset(data, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype> << <DLEX_GET_BLOCKS(N), DLEX_CUDA_NUM_THREADS >> >(
    N, alpha, data);
}
template void set_gpu<int>(const int N, const int alpha, int* data);
template void set_gpu<float>(const int N, const float alpha, float* data);
template void set_gpu<double>(const int N, const double alpha, double* data);

template <typename Dtype>
__global__ void div_inplace_kernel(const int n, const Dtype alpha, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data[index] /= alpha;
  }
}
template <typename Dtype>
void div_inplace_gpu(const int N, const Dtype alpha, Dtype* data) {
  div_inplace_kernel<Dtype> << <DLEX_GET_BLOCKS(N), DLEX_CUDA_NUM_THREADS >> >(
    N, alpha, data);
}
template void div_inplace_gpu<int>(const int N, const int alpha, int* data);
template void div_inplace_gpu<float>(const int N, const float alpha, float* data);
template void div_inplace_gpu<double>(const int N, const double alpha, double* data);

template <>
void dlex_gpu_rng_gaussian(const int n, const float mu, const float sigma, float* r) {
  CURAND_DCHECK(curandGenerateNormal(CuHandleManager::curand_generator(), r, n, mu, sigma));
}

template <>
void dlex_gpu_rng_gaussian(const int n, const double mu, const double sigma, double* r) {
  CURAND_DCHECK(curandGenerateNormalDouble(CuHandleManager::curand_generator(), r, n, mu, sigma));
}

template <>
void gemm_gpu<float>(cublasHandle_t cublas_handle, const bool TransA,
  const bool TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_DCHECK(cublasSgemm(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm_gpu<double>(cublasHandle_t cublas_handle, const bool TransA,
  const bool TransB, const int M, const int N, const int K,
  const double alpha, const double* A, const double* B, const double beta,
  double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == false) ? K : M;
  int ldb = (TransB == false) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == false) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_DCHECK(cublasDgemm(cublas_handle, cuTransB, cuTransA,
    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

///////////////////// add_bias ///////////////////////
template <typename Dtype>
__global__ void add_bias_kernel(const int n, const Dtype* bias, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, n) {
    dst[index] += bias[index];
  }
}

template <typename Dtype>
void add_bias_gpu(const int num, const int len, const Dtype* bias, Dtype* dst) {
  for (int n = 0; n < num; n++) {
    Dtype* dst_n = dst + n * len;
    add_bias_kernel<Dtype> << <DLEX_GET_BLOCKS(len), DLEX_CUDA_NUM_THREADS >> >(
      len, bias, dst_n);
  }
}
template void add_bias_gpu<float>(const int num, const int len, const float* bias, float* dst);
template void add_bias_gpu<double>(const int num, const int len, const double* bias, double* dst);

template <typename Dtype>
__global__ void add_bias_kernel(const int n, const int len, const Dtype* bias, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype b = bias[index];
    Dtype *dst_index = dst + index * len;
    for (int k = 0; k < len; k++)
      dst_index[k] += b;
  }
}

template <typename Dtype>
void add_bias_gpu(const int num, const int ch_size, const int len, const Dtype* bias, Dtype* dst) {
  for (int n = 0; n < num; n++) {
    Dtype* dst_n = dst + n * ch_size * len;
    add_bias_kernel<Dtype> << <DLEX_GET_BLOCKS(ch_size), DLEX_CUDA_NUM_THREADS >> >(
      ch_size, len, bias, dst_n);
  }
}
template void add_bias_gpu<float>(const int num, const int ch_size, const int len, const float* bias, float* dst);
template void add_bias_gpu<double>(const int num, const int ch_size, const int len, const double* bias, double* dst);


///////////////////// backward_bias ///////////////////////
template <typename Dtype>
__global__ void backward_bias_kernel(const int n, Dtype* next_diff_n, Dtype* bias_gradient) {
  CUDA_KERNEL_LOOP(index, n) {
    bias_gradient[index] += 1.0f * next_diff_n[index];
  }
}

template <typename Dtype>
void backward_bias_gpu(const int num, const int len, Dtype* next_diff, Dtype* bias_gradient) {
  for (int n = 0; n < num; n++) {
    Dtype* next_diff_n = next_diff + n * len;
    backward_bias_kernel<Dtype> << <DLEX_GET_BLOCKS(len), DLEX_CUDA_NUM_THREADS >> >(
      len, next_diff_n, bias_gradient);
  }
}
template void backward_bias_gpu<float>(const int num, const int len, float* next_diff, float* bias_gradient);
template void backward_bias_gpu<double>(const int num, const int len, double* next_diff, double* bias_gradient);

template <typename Dtype>
__global__ void backward_bias_kernel(const int n, const int len, Dtype* next_diff_n, Dtype* bias_gradient) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype b = 0;
    Dtype* next_diff_nl = next_diff_n + index*len;
    for (int k = 0; k < len; k++)
    {
      b += 1.0f*next_diff_nl[k];
    }
    bias_gradient[index] += b;
  }
}

template <typename Dtype>
void backward_bias_gpu(const int num, const int ch_size, const int len, Dtype* next_diff, Dtype* bias_gradient) {
  for (int n = 0; n < num; n++) {
    Dtype* next_diff_n = next_diff + n * ch_size * len;
    backward_bias_kernel<Dtype> << <DLEX_GET_BLOCKS(ch_size), DLEX_CUDA_NUM_THREADS >> >(
      ch_size, len, next_diff_n, bias_gradient);
  }
}
template void backward_bias_gpu<float>(const int num, const int ch_size, const int len, float* next_diff, float* bias_gradient);
template void backward_bias_gpu<double>(const int num, const int ch_size, const int len, double* next_diff, double* bias_gradient);

///////////////////// im2col ///////////////////////
template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int height_col, const int width_col,
  Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype> << <DLEX_GET_BLOCKS(num_kernels),
    DLEX_CUDA_NUM_THREADS >> >(
    num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
    pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
    width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w, double* data_col);

///////////////////// col2im ///////////////////////
template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
  const int height, const int width, const int channels,
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int height_col, const int width_col,
  Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
            height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h,
  const int stride_w, const int dilation_h, const int dilation_w,
  Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
    stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
    stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype> << <DLEX_GET_BLOCKS(num_kernels),
    DLEX_CUDA_NUM_THREADS >> >(
    num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
    height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h,
  const int stride_w, const int dilation_h, const int dilation_w,
  float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h,
  const int stride_w, const int dilation_h, const int dilation_w,
  double* data_im);
}
#endif
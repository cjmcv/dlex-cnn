#ifdef USE_CUDA
#include "util/device.h"
#include "util/math_functions.h"

namespace dlex_cnn
{
	template <typename Dtype>
	__global__ void set_kernel(const int n, const Dtype alpha, Dtype* data)
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			data[index] = alpha;
		}
	}

	template <typename Dtype>
	void dlex_gpu_set(const int N, const Dtype alpha, Dtype* data) 
	{
		if (alpha == 0) 
		{
			CUDA_DCHECK(cudaMemset(data, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
			return;
		}
		// NOLINT_NEXT_LINE(whitespace/operators)
		set_kernel<Dtype> << <DLEX_GET_BLOCKS(N), DLEX_CUDA_NUM_THREADS >> >(
			N, alpha, data);
	}
	template void dlex_gpu_set<int>(const int N, const int alpha, int* data);
	template void dlex_gpu_set<float>(const int N, const float alpha, float* data);
	template void dlex_gpu_set<double>(const int N, const double alpha, double* data);

	template <typename Dtype>
	__global__ void div_inplace_kernel(const int n, const Dtype alpha, Dtype* data)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			data[index] /= alpha;
		}
	}
	template <typename Dtype>
	void div_inplace_gpu(const int N, const Dtype alpha, Dtype* data)
	{
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
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));//Caffe::Layer<Dtype>::cublas_handle()
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
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));//Caffe::Layer<Dtype>::cublas_handle()
	}
}
#endif
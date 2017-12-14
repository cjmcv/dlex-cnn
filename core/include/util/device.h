////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DEVICE_HPP_
#define DLEX_DEVICE_HPP_

#ifdef USE_CUDA

#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cublas_v2.h>
#include <ctime>

namespace dlex_cnn
{
	const char* cublasGetErrorString(cublasStatus_t error);
	const char* curandGetErrorString(curandStatus_t error);

	// CUDA: use 512 threads per block
	const int DLEX_CUDA_NUM_THREADS = 512;

	// CUDA: number of blocks for threads.
	inline int DLEX_GET_BLOCKS(const int N) {
		return (N + DLEX_CUDA_NUM_THREADS - 1) / DLEX_CUDA_NUM_THREADS;
	}

	// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_DCHECK(condition) \
	do { \
		cudaError_t error = condition; \
		if(error != cudaSuccess) { \
			DLOG_ERR("CUDA_DCHECK: %s", cudaGetErrorString(error)); \
			throw(CUDA_DCHECK_EXC);	\
		}\
	} while (0)

#define CURAND_DCHECK(condition) \
	do { \
		curandStatus_t status = condition; \
		if(status != CURAND_STATUS_SUCCESS) { \
			DLOG_ERR("CURAND_DCHECK: %s", curandGetErrorString(status)); \
			throw(CURAND_DCHECK_EXC);	\
		} \
	} while (0)

#define CUBLAS_DCHECK(condition) \
	do { \
		cublasStatus_t status = condition; \
		if(status != CUBLAS_STATUS_SUCCESS) { \
			DLOG_ERR("CUBLAS_DCHECK: %s", cublasGetErrorString(status)); \
			throw(CUBLAS_DCHECK_EXC);	\
		} \
    } while (0)

	// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_DCHECK(cudaPeekAtLastError())

	// curand
	class CuHandleManager
	{
	public:
		CuHandleManager();
		~CuHandleManager();

		static CuHandleManager& Get();
		inline static cublasHandle_t cublas_handle() {
			return Get().cublas_handle_;
		}
		inline static curandGenerator_t curand_generator() {
			return Get().curand_generator_;
		}

	private:
		long long seedgen();

		cublasHandle_t cublas_handle_;
		curandGenerator_t curand_generator_;
	};
}
#endif	//USE_CUDA

#endif //DLEX_COMMON_HPP_

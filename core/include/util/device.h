////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DEVICE_HPP_
#define DLEX_DEVICE_HPP_

#ifndef CPU_ONLY

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

namespace dlex_cnn
{
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

#define DCUDA_CHECK(condition) \
	do { \
		cudaError_t error = condition; \
		if(error != cudaSuccess) { \
			fprintf(stderr, cudaGetErrorString(error)); \
			throw(1);	\
		}\
	} while (0)

//#define CURAND_CHECK(condition) \
//	do { \
//		curandStatus_t status = condition; \
//		DCHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
//	      << curandGetErrorString(status); \
//	} while (0)

}
#endif	//CPU_ONLY

#endif //DLEX_COMMON_HPP_
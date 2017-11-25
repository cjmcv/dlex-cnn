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

namespace dlex_cnn
{

#define DCUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
	if(error != cudaSuccess) { \
		fprintf(stderr, cudaGetErrorString(error)); \
		throw(1);	\
			}\
      } while (0)

}
#endif	//CPU_ONLY

#endif //DLEX_COMMON_HPP_
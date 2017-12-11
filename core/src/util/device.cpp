////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA

#include "util/device.h"

namespace dlex_cnn
{
	static CuHandleManager *gHandleManager = NULL;
	CuHandleManager& CuHandleManager::Get()
	{
		if (gHandleManager == NULL)
			gHandleManager = new CuHandleManager();
		return *gHandleManager;
	}

	CuHandleManager::CuHandleManager() :
		cublas_handle_(NULL), curand_generator_(NULL)
	{
		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
			DLOG_ERR("Cannot create Cublas handle. Cublas won't be available.");
		}
		// Try to create a curand handler.
		if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS
			|| curandSetPseudoRandomGeneratorSeed(curand_generator_, seedgen()) != CURAND_STATUS_SUCCESS)
			DLOG_ERR("Cannot create Curand generator. Curand won't be available.");
	}

	CuHandleManager::~CuHandleManager()
	{
		if (cublas_handle_)
			CUBLAS_DCHECK(cublasDestroy(cublas_handle_));
		if (curand_generator_)
			CURAND_DCHECK(curandDestroyGenerator(curand_generator_));
	}

	long long CuHandleManager::seedgen()
	{
		long long s, seed;
		s = time(NULL);
		seed = std::abs(((s * 181) * ((s - 83) * 359)) % 104729);
		return seed;
	}
}
#endif
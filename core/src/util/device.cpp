////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  A common file for cuda.
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA

#include "util/device.h"

namespace dlex_cnn
{
	// One object is enough for one thread in one gpu.
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
		// Try to create a cublas handler.
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

	// Generate a seed for curand.
	long long CuHandleManager::seedgen()
	{
		long long s, seed;
		s = time(NULL);
		seed = std::abs(((s * 181) * ((s - 83) * 359)) % 104729);
		return seed;
	}

	// Error string
	const char* cublasGetErrorString(cublasStatus_t error) {
		switch (error) {
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
		}
		return "Unknown cublas status";
	}

	const char* curandGetErrorString(curandStatus_t error) {
		switch (error) {
		case CURAND_STATUS_SUCCESS:
			return "CURAND_STATUS_SUCCESS";//< No errors
		case CURAND_STATUS_VERSION_MISMATCH:
			return "CURAND_STATUS_VERSION_MISMATCH";//< Header file and linked library version do not match
		case CURAND_STATUS_NOT_INITIALIZED:
			return "CURAND_STATUS_NOT_INITIALIZED";//< Generator not initialized
		case CURAND_STATUS_ALLOCATION_FAILED:
			return "CURAND_STATUS_ALLOCATION_FAILED";//< Memory allocation failed
		case CURAND_STATUS_TYPE_ERROR:
			return "CURAND_STATUS_TYPE_ERROR";//< Generator is wrong type
		case CURAND_STATUS_OUT_OF_RANGE:
			return "CURAND_STATUS_OUT_OF_RANGE";//< Argument out of range
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
			return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";//< Length requested is not a multple of dimension
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";//< GPU does not have double precision required by MRG32k3a
		case CURAND_STATUS_LAUNCH_FAILURE:
			return "CURAND_STATUS_LAUNCH_FAILURE";//< Kernel launch failure
		case CURAND_STATUS_PREEXISTING_FAILURE:
			return "CURAND_STATUS_PREEXISTING_FAILURE";//< Preexisting failure on library entry
		case CURAND_STATUS_INITIALIZATION_FAILED:
			return "CURAND_STATUS_INITIALIZATION_FAILED";//< Initialization of CUDA failed
		case CURAND_STATUS_ARCH_MISMATCH:
			return "CURAND_STATUS_ARCH_MISMATCH";//< Architecture mismatch, GPU does not support requested feature
		case CURAND_STATUS_INTERNAL_ERROR:
			return "CURAND_STATUS_INTERNAL_ERROR";//< Internal library error
		}
		return "Unknown curand status";
	}
}
#endif
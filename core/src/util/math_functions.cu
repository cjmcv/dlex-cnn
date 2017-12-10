#include "util/device.h"

namespace dlex_cnn
{
#ifdef USE_CUDA
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
#endif
}
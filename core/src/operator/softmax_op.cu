////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Softmax.
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include "operator/softmax_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	__global__ void FindMax(const int n, const Dtype* arr, Dtype &max_val) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			int tid = threadIdx.x;
			extern __shared__ float shared_data[];
			shared_data[tid] = arr[index];
			__syncthreads();

			for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
			{
				if (tid<s)
					shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
				__syncthreads();
			}

			if (tid == 0)
				max_val = shared_data[tid];	//max_val[blockIdx.x] = shared_data[tid];	need to reduce
		}
	}
	template <typename Dtype>
	__global__ void SubExp(const int n, const Dtype* in_data, const Dtype val, Dtype* out_data) {
		CUDA_KERNEL_LOOP(index, n) {
			out_data[index] = exp(in_data[index] - val);
		}
	}
	template <typename Dtype>
	__global__ void GetSum(const int n, const Dtype* arr, Dtype &sum)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			int tid = threadIdx.x;
			extern __shared__ float shared_data[];
			shared_data[tid] = arr[index];
			__syncthreads();

			for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
			{
				if (tid<s)
					shared_data[tid] += shared_data[tid + s];
				__syncthreads();
			}

			if (tid == 0)
				sum = shared_data[tid];
		}
	}
	template <typename Dtype>
	__global__ void DivInplace(const int n, const Dtype val, Dtype* data) {
		CUDA_KERNEL_LOOP(index, n) {
			data[index] = data[index] / val;
		}
	}
	template <typename Dtype>
	void SoftmaxOp<Dtype>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();
		//const std::vector<int> prev_data_shape = prev[0]->getShape();
		const std::vector<int> next_data_shape = next[0]->getShape();

		Dtype *prev_data_base = (Dtype *)prev[0]->getPushGpuData();
		Dtype *next_data_base = (Dtype *)next[0]->getPushGpuData();

		const int next_data_num = next_data_shape[tind::eNum];
		const int prev_data_size3D = prev_data_size[tind::e3D];
		const int next_data_size3D = next_data_size[tind::e3D];

		next[0]->setGpuZero();

		for (int nn = 0; nn < next_data_num; nn++)
		{
			const Dtype* prev_data = prev_data_base + nn * prev_data_size3D;
			Dtype* next_data = next_data_base + nn * next_data_size3D;

			//step1 : find max value
			Dtype max_val = prev_data[0];
			FindMax<Dtype> << <DLEX_GET_BLOCKS(prev_data_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_data_size3D, prev_data, max_val);

			//step2 : sum
			Dtype sum = 0;
			SubExp<Dtype> << <DLEX_GET_BLOCKS(prev_data_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_data_size3D, prev_data, max_val, next_data);
			GetSum<Dtype> << <DLEX_GET_BLOCKS(prev_data_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_data_size3D, next_data, sum);

			//step3 : div
			DivInplace<Dtype> << <DLEX_GET_BLOCKS(prev_data_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_data_size3D, sum, next_data);
		}

		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void SoftmaxBackwardKernel1(const int n, const Dtype* next_data, const Dtype* next_diff, Dtype *prev_diff)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			const Dtype val_next_data = next_data[index];
			Dtype val = prev_diff[index];
			for (int next_diff_idx = 0; next_diff_idx < n; next_diff_idx++)
			{
				val -= val_next_data * next_data[next_diff_idx] * next_diff[next_diff_idx];
			}
			prev_diff[index] = val;
		}
	}
	template <typename Dtype>
	__global__ void SoftmaxBackwardKernel2(const int n, const Dtype* next_data, const Dtype* next_diff, Dtype *prev_diff)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			Dtype val = next_data[index] * next_diff[index];
			prev_diff[index] += val;
		}
	}
	template <typename Dtype>
	void SoftmaxOp<Dtype>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff
		)
	{
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();
		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();

		const std::vector<int> prev_data_shape = prev[0]->getShape();
		const std::vector<int> next_data_shape = next[0]->getShape();
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		Dtype *prev_data_base = (Dtype *)prev[0]->getPushGpuData();
		Dtype *next_data_base = (Dtype *)next[0]->getPushGpuData();
		Dtype *prev_diff_base = (Dtype *)prev_diff[0]->getPushGpuData();
		Dtype *next_diff_base = (Dtype *)next_diff[0]->getPushGpuData();

		if (prev_data_size[tind::e4D] != next_data_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input and output data must be equal \n");
			return;
		}
		if (prev_diff_size[tind::e4D] != next_diff_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output diff must be equal \n");
			return;
		}
		if (prev_diff_size[tind::e4D] != prev_data_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output data must be equal \n");
			return;
		}

		//update prev_diff
		prev_diff[0]->setGpuZero();
		const int prev_data_size3D = prev_data_size[tind::e3D];
		const int next_data_size3D = next_data_size[tind::e3D];
		const int prev_diff_size3D = prev_diff_size[tind::e3D];
		const int next_diff_size3D = next_diff_size[tind::e3D];
		for (int pn = 0; pn < prev_data_shape[tind::eNum]; pn++)
		{
			const Dtype* prev_data = prev_data_base + pn * prev_data_size3D;
			const Dtype* next_data = next_data_base + pn * next_data_size3D;
			const Dtype* next_diff = next_diff_base + pn * next_diff_size3D;
			Dtype* prev_diff = prev_diff_base + pn * prev_diff_size3D;

			SoftmaxBackwardKernel1<Dtype> << <DLEX_GET_BLOCKS(prev_diff_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_diff_size3D, next_data, next_diff, prev_diff);

			SoftmaxBackwardKernel2<Dtype> << <DLEX_GET_BLOCKS(prev_diff_size3D), DLEX_CUDA_NUM_THREADS >> >(
				prev_diff_size3D, next_data, next_diff, prev_diff);
		}
	}
	template void SoftmaxOp<float>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<float>>> &prev,
		const std::vector<std::shared_ptr<Tensor<float>>> &next);
	template void SoftmaxOp<double>::forward_gpu(
		const std::vector<std::shared_ptr<Tensor<double>>> &prev,
		const std::vector<std::shared_ptr<Tensor<double>>> &next);
	template void SoftmaxOp<float>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<float>>> &prev,
		const std::vector<std::shared_ptr<Tensor<float>>> &next,
		const std::vector<std::shared_ptr<Tensor<float>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<float>>> &next_diff);
	template void SoftmaxOp<double>::backward_gpu(
		const std::vector<std::shared_ptr<Tensor<double>>> &prev,
		const std::vector<std::shared_ptr<Tensor<double>>> &next,
		const std::vector<std::shared_ptr<Tensor<double>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<double>>> &next_diff);
}//namespace
#endif
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Softmax.
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include "operator/softmax_op.h"
#include <sstream>
#include <float.h>

namespace dlex_cnn
{
	template <typename Dtype>
	__global__ void MaxPerNum(const int num, const int size, Dtype* arr, Dtype* out)
	{
		CUDA_KERNEL_LOOP(index, num)
		{
			Dtype *base = arr + index * size;
			Dtype maxval = -FLT_MAX;
			for (int i = 0; i < size; i++)
				maxval = max(maxval, base[i]);
			out[index] = maxval;
		}
	}
	template <typename Dtype>
	__global__ void SubExpPerNum(const int num, const int size, Dtype* in_data, const Dtype* val, Dtype* out_data) {
		CUDA_KERNEL_LOOP(index, num) {
			Dtype *in_base = in_data + index * size;
			Dtype *out_base =out_data + index * size;
			for (int i = 0; i < size; i++)
				out_base[i] = exp(in_base[i] - val[index]);
		}
	}
	template <typename Dtype>
	__global__ void SumPerNum(const int n, const int size, Dtype* arr, Dtype* out)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			Dtype *base = arr + index * size;
			Dtype sum = 0;
			for (int i = 0; i < size; i++)
				sum += base[i];
			out[index] = sum;
		}
	}
	template <typename Dtype>
	__global__ void DivInplacePerNum(const int num, const int size, const Dtype *val, Dtype* data) {
		CUDA_KERNEL_LOOP(index, num) {
			Dtype *arr = data + index * size;
			for (int i = 0; i < size; i++)
				arr[i] = arr[i] / val[index];
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
		Dtype *next_data_base = (Dtype *)next[0]->getGpuData();

		const int next_data_num = next_data_shape[tind::eNum];
		const int prev_data_size3D = prev_data_size[tind::e3D];
		const int next_data_size3D = next_data_size[tind::e3D];

		next[0]->setGpuZero();

		if (gpu_num_temp_ == NULL)
			CUDA_DCHECK(cudaMalloc(&gpu_num_temp_, sizeof(Dtype) * next_data_num));

		CUDA_DCHECK(cudaMemset(gpu_num_temp_, 0, sizeof(Dtype) * next_data_num));

		MaxPerNum<Dtype> << <DLEX_GET_BLOCKS(next_data_num), DLEX_CUDA_NUM_THREADS >> >(
			next_data_num, prev_data_size3D, prev_data_base, gpu_num_temp_);

		SubExpPerNum<Dtype> << <DLEX_GET_BLOCKS(next_data_num), DLEX_CUDA_NUM_THREADS >> >(
			next_data_num, prev_data_size3D, prev_data_base, gpu_num_temp_, next_data_base);
		SumPerNum<Dtype> << <DLEX_GET_BLOCKS(next_data_num), DLEX_CUDA_NUM_THREADS >> >(
			next_data_num, prev_data_size3D, next_data_base, gpu_num_temp_);

		DivInplacePerNum<Dtype> << <DLEX_GET_BLOCKS(next_data_num), DLEX_CUDA_NUM_THREADS >> >(
			next_data_num, prev_data_size3D, gpu_num_temp_, next_data_base);

		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void SoftmaxBackwardKernel1(const int n, const Dtype* next_data, const Dtype* next_diff, Dtype *prev_diff)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			const Dtype val_next_data = next_data[index];
			Dtype val_prev_diff = prev_diff[index];
			for (int next_diff_idx = 0; next_diff_idx < n; next_diff_idx++)
			{
				val_prev_diff -= val_next_data * next_data[next_diff_idx] * next_diff[next_diff_idx];
			}
			prev_diff[index] = val_prev_diff;
		}
	}
	template <typename Dtype>
	__global__ void SoftmaxBackwardKernel2(const int n, const Dtype* next_data, const Dtype* next_diff, Dtype *prev_diff)
	{
		CUDA_KERNEL_LOOP(index, n) {
			prev_diff[index] += next_data[index] * next_diff[index];
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
		Dtype *prev_diff_base = (Dtype *)prev_diff[0]->getGpuData();
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
			
			CUDA_POST_KERNEL_CHECK;
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

////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/device.h"
#include "optimizer/optimizer.h"

namespace dlex_cnn
{
	//SGD, w -= lr*g 
	template <typename Dtype>
	__global__ void sgd_update_kernel(const int n, const float lr, const Dtype* gradient, Dtype* weight)
	{
		CUDA_KERNEL_LOOP(index, n)
		{
			weight[index] -= lr * gradient[index];
		}
	}
	template <typename Dtype>
	void SGD<Dtype>::update_gpu(std::shared_ptr< Node<Dtype> > node)
	{
		const std::vector<std::shared_ptr<Tensor<Dtype>>> node_data = node->getDataVec();
		if (node_data.size() == 1)
			return;

		const std::vector<std::shared_ptr<Tensor<Dtype>>> op_data = node->getInteOp()->getOpGradient();

		Dtype* weight_data = (Dtype *)node_data[1]->getPushGpuData();
		const std::vector<int> weight_data_size = node_data[1]->getSize();
		const Dtype* w_gradient_data = (Dtype *)op_data[0]->getPushGpuData();
		int N = weight_data_size[tind::e4D];
		sgd_update_kernel<Dtype> << <DLEX_GET_BLOCKS(N), DLEX_CUDA_NUM_THREADS >> >(
			N, Optimizer<Dtype>::lr_,
			w_gradient_data, weight_data);

		if (node_data.size() >= 2 && op_data.size() >= 2)
		{
			Dtype* blas_data = (Dtype *)node_data[2]->getPushGpuData();
			const std::vector<int> blas_data_size = node_data[2]->getSize();
			const Dtype* b_gradient_data = (Dtype *)op_data[1]->getPushGpuData();
			N = blas_data_size[tind::e4D];
			sgd_update_kernel<Dtype> << <DLEX_GET_BLOCKS(N), DLEX_CUDA_NUM_THREADS >> >(
				N, Optimizer<Dtype>::lr_,
				b_gradient_data, blas_data);
		}
	}
	template void SGD<float>::update_gpu(std::shared_ptr< Node<float> > node);
	template void SGD<double>::update_gpu(std::shared_ptr< Node<double> > node);
}//namespace

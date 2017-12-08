////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "optimizer/optimizer.h"

namespace dlex_cnn
{
	template <typename Dtype>
	int Optimizer<Dtype>::getOptimizerByStr(std::string &type, std::shared_ptr<Optimizer<Dtype>> &opt)
	{
		if (type == "SGD")
			opt = std::shared_ptr< dlex_cnn::Optimizer<Dtype> >(new dlex_cnn::SGD<Dtype>());
		else
			return -1;

		return 0;
	}
	INSTANTIATE_CLASS(Optimizer);
    
	//SGD
	//w -= lr*g
	template <typename Dtype>
	void SGD<Dtype>::update(std::shared_ptr< Node<Dtype> > node)
	{
		const std::vector<std::shared_ptr<Tensor<Dtype>>> node_data = node->getDataVec();
		if (node_data.size() == 1)
			return;

		const std::shared_ptr<Op<Dtype>> inteOp = node->getInteOp();

		Dtype* weight_data = (Dtype *)node_data[1]->getCpuData();
		const std::vector<int> weight_data_size = node_data[1]->getSize();
		const Dtype* w_gradient_data = (Dtype *)(inteOp->getOpGradient())[0]->getCpuData();
		for (int i = 0; i < weight_data_size[tind::e4D]; i++)
			weight_data[i] -= Optimizer<Dtype>::lr_*w_gradient_data[i];

		if (node_data.size() >= 2 && inteOp->getOpGradient().size() >= 2)
		{
			Dtype* blas_data = (Dtype *)node_data[2]->getCpuData();
			const std::vector<int> blas_data_size = node_data[2]->getSize();
			const Dtype* b_gradient_data = (Dtype *)(inteOp->getOpGradient())[1]->getCpuData();
			for (int i = 0; i < blas_data_size[tind::e4D]; i++)
				blas_data[i] -= Optimizer<Dtype>::lr_*b_gradient_data[i];
		}
	}

	//INSTANTIATE_CLASS(Optimizer);
	INSTANTIATE_CLASS(SGD);
}//namespace

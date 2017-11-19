////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "optimizer/Optimizer.h"

namespace dlex_cnn
{
	template <typename Dtype>
	int Optimizer<Dtype>::getOptimizerByStr(std::string &optName, std::shared_ptr<Optimizer<Dtype>> &opt)
	{
		if (optName == "SGD")
			opt = std::shared_ptr< dlex_cnn::Optimizer<Dtype> >(new dlex_cnn::SGD<Dtype>());
		else
			return -1;

		return 0;
	}
	//SGD
	//w -= lr*g
	template <typename Dtype>
	void SGD<Dtype>::update(std::shared_ptr<Node<Dtype>> node)
	{
		const std::vector<std::shared_ptr<Tensor<Dtype>>> nodeData = node->getDataVec();
		if (nodeData.size() == 1)
			return;

		const std::shared_ptr<Op<Dtype>> inteOp = node->getInteOp();

		Dtype* weightData = (Dtype *)nodeData[1]->getData();
		const std::vector<int> weightDataSize = nodeData[1]->getSize();
		const Dtype* wGradientData = (Dtype *)(inteOp->getOpGradient())[0]->getData();
		for (int i = 0; i < weightDataSize[tind::e4D]; i++)
			weightData[i] -= lr_*wGradientData[i];

		if (nodeData.size() >= 2 && inteOp->getOpGradient().size() >= 2)
		{
			Dtype* blasData = (Dtype *)nodeData[2]->getData();
			const std::vector<int> blasDataSize = nodeData[2]->getSize();
			const Dtype* bGradientData = (Dtype *)(inteOp->getOpGradient())[1]->getData();
			for (int i = 0; i < blasDataSize[tind::e4D]; i++)
				blasData[i] -= lr_*bGradientData[i];
		}
	}

	INSTANTIATE_CLASS(Optimizer);
	INSTANTIATE_CLASS(SGD);
}//namespace
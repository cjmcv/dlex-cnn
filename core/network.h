////////////////////////////////////////////////////////////////
// > Copyright (c) 2017, Jianming Chen. All rights reserved. 
// > https://github.com/cjmcv/cjmcv.github.io
// > file   network.h
// > brief  The Integration of each module
// > date   2017.07.18
////////////////////////////////////////////////////////////////

#ifndef DLEX_NETWORK_HPP_
#define DLEX_NETWORK_HPP_

#include <memory>
#include <vector>
#include "configure.h"
#include "optimizer/optimizer.h"

#include "graph.h"
#include "node.h"
#include "tensor.h"

namespace dlex_cnn 
{
	template <typename Dtype>
	class NetWork
	{
	public:
		NetWork();
		virtual ~NetWork();
	public:
		int netWorkInit(std::string name);

		int saveBinModel(const std::string &modelFile);
		int loadBinModel(const std::string &modelFile);
		
		int saveStageModel(const std::string &path, const int stage);	
		int readHyperParams(FILE *fp);
		int loadStageModel(const std::string &path, const int stage);

		std::shared_ptr<Tensor<Dtype>> testBatch(const std::shared_ptr<Tensor<Dtype>> inputDataTensor, const std::shared_ptr<Tensor<Dtype>> labelDataTensor = NULL);
	
		void setOptimizer(std::shared_ptr<Optimizer<Dtype>> optimizer);
		void setLearningRate(const float lr);

		float trainBatch(const std::shared_ptr<Tensor<Dtype>> inputDataTensor,
			const std::shared_ptr<Tensor<Dtype>> labelDataTensor);
		int getNodeData(const std::string &nodeName, std::shared_ptr<Tensor<Dtype>> &cpuData);
		inline const std::shared_ptr<Graph<Dtype>> getGraph() { return graph_; };

		void addNode(std::string &nodeName, 
			std::vector<std::shared_ptr<Op<Dtype>>> &op, 
			std::vector<std::string> &inNodeNames = std::vector<std::string>());
		int switchPhase(int phase);

		// fill the input data and label date (during training), then compute graph forward
		int forward(const std::shared_ptr<Tensor<Dtype>> inputDataTensor, const std::shared_ptr<Tensor<Dtype>> labelDataTensor = NULL);
		// compute graph backward and update nodes'paramaters
		int backward();

		int netWorkShow();
		
	private:
		// Train/Test, should be the same as graph's
		int phase_ = tind::Train;
		std::string name_;
		// Mainly contains nodes and operators
		std::shared_ptr<Graph<Dtype>> graph_;
		// Optimizer to update node's paramater during training
		std::shared_ptr<Optimizer<Dtype>> optimizer_;
	};

}
#endif DLEX_NETWORK_HPP_
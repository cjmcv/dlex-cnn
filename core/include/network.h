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

#include "task.h"
#include "graph.h"
#include "node.h"
#include "tensor.h"
#include "io/data_prefetcher.h"

namespace dlex_cnn 
{
	template <typename Dtype>
	class NetWork
	{
	public:
		NetWork(std::string name);
		virtual ~NetWork();
	public:
		DataPrefetcher<Dtype> prefetcher_;

		int netParamsInit();

		int saveBinModel(const std::string &model_file);
		int loadBinModel(const std::string &model_file);
		
		int saveStageModel(const std::string &path, const int stage);	
		int readHyperParams(FILE *fp);
		int loadStageModel(const std::string &path, const int stage);

		std::shared_ptr<Tensor<Dtype>> testBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor, const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);
	
		void setOptimizer(std::shared_ptr<Optimizer<Dtype>> optimizer);
		void setLearningRate(const float lr);

		inline int setIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names);
		int feedDataByPrefetcher();
		float trainBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor = NULL, 
			const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);
		int getNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &cpuData);
		inline const std::shared_ptr<Graph<Dtype>> getGraph() { return graph_; };

		void addNode(const std::string &node_name, 
			const std::vector<std::shared_ptr<Op<Dtype>>> &op, 
			const std::vector<std::string> &in_node_names = std::vector<std::string>());
		int switchPhase(int phase);

		// fill the input data and label date for training first, then compute graph forward
		int forward(const std::shared_ptr<Tensor<Dtype>> input_data_tensor = NULL, const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);
		// compute graph backward and update nodes'paramaters
		int backward();

		int netWorkShow();
		
	private:
		int device_id_;
		// Train/Test, should be the same as graph's
		int phase_ = tind::Train;
		std::string name_;

		// Mainly contains nodes and operators
		std::shared_ptr<Graph<Dtype>> graph_;
		// Optimizer to update node's paramater during training
		std::shared_ptr<Optimizer<Dtype>> optimizer_;
		
		// As the intermediate variable to transfer input/label data.
		std::vector<std::shared_ptr<Tensor<Dtype>>> input_data_vec_;
		std::vector<std::shared_ptr<Tensor<Dtype>>> label_data_vec_;
	};

}
#endif DLEX_NETWORK_HPP_

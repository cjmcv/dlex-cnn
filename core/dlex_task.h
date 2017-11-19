////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TASK_HPP_
#define DLEX_TASK_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include "network.h"
#include "node.h"

namespace dlex_cnn
{
	template <typename Dtype>
	class Task
	{
	public:
		Task();
		virtual ~Task();
	public:
		std::vector<NetWork> netWorks;
		std::vector<Node<Dtype>> nodes;		//中间节点，用于网络间通信？未完成，最后处理
		
		//std::vector<Node>

		//float getLoss(const std::shared_ptr<DataTensor> labelDataTensor, const std::shared_ptr<DataTensor> outputDataTensor);

	//private:
	//	void setPhase(Phase phase);

	};
}

#endif
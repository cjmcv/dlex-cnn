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
#include "io/data_prefetcher.h"

namespace dlex_cnn
{
	class Task
	{
	public:
		Task();
		virtual ~Task();
	public:
		tind::Mode device_mode_;
		//DataPrefetcher<Dtype> prefetcher_;

		//static Task& Get();
		//inline static tind::Mode mode() { return Get().device_mode_; }
		//inline static tind::Mode prefetcher() { return Get().prefetcher_; }

		//std::vector<std::shared_ptr<NetWork<Dtype>>> networks_;		// Only support one network for now.
		//std::vector<std::shared_ptr<Node<Dtype>>> net_nodes_;	// Intermediate nodes of different networks, NOT IMPLEMENTED!
	};
}

#endif
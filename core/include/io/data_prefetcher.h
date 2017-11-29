////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

// 多线程数据预存, batch中暂先使用内存池
#ifndef DLEX_PREFETCH_HPP_
#define DLEX_PREFETCH_HPP_

#include "util/thread_pool.h"
#include "tensor.h"

#include <queue>

namespace dlex_cnn
{
	template <typename Dtype>
	class DataPrefetcher
	{
	public:
		DataPrefetcher();
		virtual ~DataPrefetcher();

	// 设备信息监管全局，prefetch由input_op取数，外部入队列（单独开一线程自己处理）。
	// 提取mnist数据，pushdata（判断模式，若为GPU则入到GPU队列），input_op（prefetcher作为入参输入到input，prefetcher属于network）内pushData到下一层
	// block_queue
	public:
		int pushData(std::shared_ptr<dlex_cnn::Tensor<Dtype>> input_data,
			std::shared_ptr<dlex_cnn::Tensor<Dtype>> label_data);
		int popData();

	protected:
		void entryInnerThread();

	private:
		std::queue < std::pair < std::shared_ptr<dlex_cnn::Tensor<float>>, std::shared_ptr<dlex_cnn::Tensor<float>> > > data_queue_;
		std::vector<float> mean_value_;
	};
}

#endif //DLEX_PREFETCH_HPP_
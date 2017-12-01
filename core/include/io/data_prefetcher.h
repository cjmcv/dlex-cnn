////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

// 多线程数据预存, batch中暂先使用内存池
#ifndef DLEX_PREFETCH_HPP_
#define DLEX_PREFETCH_HPP_

#include "tensor.h"
#include "util/thread_inner.h"
#include "util/blocking_queue.h"

#include <queue>

namespace dlex_cnn
{
	template <typename Dtype>
	class DataPrefetcher : ThreadInner
	{
		using TensorPair = std::pair < std::shared_ptr<Tensor<Dtype>>, std::shared_ptr<Tensor<Dtype>> >;
	public:
		DataPrefetcher();
		virtual ~DataPrefetcher();

	// 设备信息监管全局，prefetch由input_op取数，外部入队列（单独开一线程自己处理）。
	// 提取mnist数据，pushdata（判断模式，若为GPU则入到GPU队列），input_op（prefetcher作为入参输入到input，prefetcher属于network）内pushData到下一层
	// block_queue
	public:
		static const int PREFETCH_COUNT = 3;

		inline void setInstantiation(void *ptr) { instant_ = ptr; }
		void *instant_ = NULL;
		bool(*batch_loader_pfunc_)(void *, TensorPair*) = NULL;
		bool loadBatch(TensorPair* batch);

		virtual void entryInnerThread();
	private:

		TensorPair base_storage_[PREFETCH_COUNT];
		BlockingQueue < TensorPair* > free_;
		BlockingQueue < TensorPair* > full_;
	};
}

#endif //DLEX_PREFETCH_HPP_
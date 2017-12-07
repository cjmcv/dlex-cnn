////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

// 多线程数据预存, batch中暂先使用内存池
#ifndef DLEX_PREFETCH_HPP_
#define DLEX_PREFETCH_HPP_

#include "task.h"
#include "tensor.h"
#include "util/thread_inner.h"
#include "util/blocking_queue.h"

#include <queue>

namespace dlex_cnn
{
	template <typename Dtype>
	class DataPrefetcher : public ThreadInner
	{
		using TensorPair = std::pair < std::shared_ptr<Tensor<Dtype>>, std::shared_ptr<Tensor<Dtype>> >;
	public:
		DataPrefetcher();
		virtual ~DataPrefetcher();

	public:
		static const int PREFETCH_COUNT = 3;

		inline void setInstantiation(void *ptr) { instant_ = ptr; }
		void *instant_ = NULL;
		bool(*batch_loader_pfunc_)(void *, TensorPair*) = NULL;
		bool loadBatch(TensorPair* batch);
		inline void feedBatchOut(TensorPair** batch) { full_.wait_and_pop(batch); }
		inline void refillBuffer(TensorPair** batch) { free_.push(*batch); }
		virtual void entryInnerThread();

	private:
		TensorPair base_storage_[PREFETCH_COUNT];
		BlockingQueue < TensorPair* > free_;
		BlockingQueue < TensorPair* > full_;
	};
}

#endif //DLEX_PREFETCH_HPP_
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Hide data copy delay between CPU and GPU.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DATA_PREFETCH_HPP_
#define DLEX_DATA_PREFETCH_HPP_

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
		// Prefetch up to 3 batches.
		static const int PREFETCH_COUNT = 3;
		// Set instance for calling the member of other class. 
		inline void setInstantiation(void *ptr) { instant_ = ptr; }
		// Batch loader, implemented by other class function.
		bool(*batch_loader_pfunc_)(void *, TensorPair*) = NULL;
		bool loadBatch(TensorPair* batch);
		// Get batch data from prefetcher to network.
		inline void feedBatchOut(TensorPair** batch) { full_.wait_and_pop(batch); }
		// Recycle buffer.
		inline void refillBuffer(TensorPair** batch) { free_.push(*batch); }
		// The entry of an inner thread, works in ThreadInner.
		virtual void entryInnerThread();

	private:
		void *instant_ = NULL;
		// Buffer for prefetch.
		TensorPair base_storage_[PREFETCH_COUNT];
		BlockingQueue < TensorPair* > free_;
		BlockingQueue < TensorPair* > full_;
	};
}

#endif //DLEX_DATA_PREFETCH_HPP_
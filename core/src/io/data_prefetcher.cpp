////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Hide data copy delay between CPU and GPU.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/device.h"
#include "io/data_prefetcher.h"

namespace dlex_cnn
{
	template <typename Dtype>
	DataPrefetcher<Dtype>::DataPrefetcher()
		: free_(), full_()
	{
		for (int i = 0; i < PREFETCH_COUNT; ++i)
			free_.push(&base_storage_[i]);
	}

	template <typename Dtype>
	DataPrefetcher<Dtype>::~DataPrefetcher()
	{
		stopInnerThread();
	}

	template <typename Dtype>
	bool DataPrefetcher<Dtype>::loadBatch(TensorPair* batch)
	{
		if (batch_loader_pfunc_ != NULL)
		{
			return batch_loader_pfunc_(instant_, batch);
		}
		else
		{
			DLOG_ERR("[ DataPrefetcher::load_batch ]: batch_loader_pfunc_ hasn't been set!");
			return false;
		}
	}

	template <typename Dtype>
	void DataPrefetcher<Dtype>::entryInnerThread()
	{
#ifdef USE_CUDA
		cudaStream_t stream;
		if (Task::mode() == tind::GPU) {
			CUDA_DCHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
		}
#endif
		try 
		{
			while (!must_stop())
			{
				TensorPair* batch;
				free_.wait_and_pop(&batch);
				if (!loadBatch(batch))
					stopInnerThread();
#ifdef USE_CUDA
				if (Task::mode() == tind::GPU) 
				{
					batch->first->asyncCpy2GPU(stream);
					batch->second->asyncCpy2GPU(stream);
					CUDA_DCHECK(cudaStreamSynchronize(stream));	//Waits for stream tasks to complete.
				}
#endif
				full_.push(batch);
			}
		}
		catch (...) {
			// Interrupted exception is expected on shutdown
		}

	}

	INSTANTIATE_CLASS(DataPrefetcher);
}

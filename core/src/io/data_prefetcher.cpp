////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

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
		try 
		{
			while (!must_stop())
			{
				TensorPair* batch;
				free_.wait_and_pop(&batch);
				if (!loadBatch(batch))
					stopInnerThread();

				full_.push(batch);
			}
		}
		catch (...) {
			// Interrupted exception is expected on shutdown
		}

	}

	INSTANTIATE_CLASS(DataPrefetcher);
}

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
	{
	}

	template <typename Dtype>
	DataPrefetcher<Dtype>::~DataPrefetcher()
	{
	}

	template <typename Dtype>
	int DataPrefetcher<Dtype>::pushData(std::shared_ptr<dlex_cnn::Tensor<Dtype>> input_data,
		std::shared_ptr<dlex_cnn::Tensor<Dtype>> label_data)
	{

	}

	template <typename Dtype>
	void DataPrefetcher<Dtype>::entryInnerThread()
	{

	}

	INSTANTIATE_CLASS(DataPrefetcher);
}

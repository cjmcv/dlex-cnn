////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DATALOADER_HPP_
#define DLEX_DATALOADER_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>

#include "tensor.h"
//#include "preprocessor.h"

namespace dlex_cnn
{
	typedef struct IMAGE_DATUM
	{
		unsigned char *pdata;
		int channels;
		int height;
		int width;
	}IMAGE_DATUM;

	template <typename Dtype>
	class ImageDataLoader
	{
	public:
		explicit ImageDataLoader(const float scale, const std::vector<float> &mean_value);
		virtual ~ImageDataLoader();

		int readImage2Tensor(const IMAGE_DATUM &image, std::shared_ptr<dlex_cnn::Tensor<Dtype>> datum, const int num_idx);

		int readVec2Tensor(const std::vector<Dtype> &vec,
			std::shared_ptr<dlex_cnn::Tensor<Dtype>> datum,
			const int num_idx);

	private:
		float scale_;
		std::vector<float> mean_value_;
	};
}

#endif //DLEX_DATALOADER_HPP_
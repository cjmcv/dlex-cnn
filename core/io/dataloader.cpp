////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dataloader.h"

namespace dlex_cnn
{
	template <typename Dtype>
	ImageDataLoader<Dtype>::ImageDataLoader(const float scale, const std::vector<float> &mean_value)
	{
		scale_ = scale;
		mean_value_ = mean_value;
	}

	template <typename Dtype>
	int ImageDataLoader<Dtype>::readImage2Tensor(const IMAGE_DATUM &image, 
		std::shared_ptr<dlex_cnn::Tensor<Dtype>> datum, 
		const int num_idx)
	{
		if (num_idx >= datum->getSize()[tind::eNum] ||
			image.channels != datum->getSize()[tind::eChannels] ||
			image.height != datum->getSize()[tind::eHeight] ||
			image.width != datum->getSize()[tind::eWidth])
		{
			//check 
			return -1;
		}
		if (scale_ <= 0 || mean_value_.size() <= 0 || mean_value_.size() != image.channels)
		{
			//check
			return -1;
		}
		if (image.channels != 1 || image.channels != 3)
		{
			//check : only support 1 or 3 channels
			return -1;
		}

		const int dataSize = datum->getSize()[tind::e3D];
		const Dtype *pdstData = (Dtype *)datum->getData() + num_idx * dataSize;

		const unsigned char *pdata = image.pdata;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				for (int c = 0; c < image.channels; c++)
				{
					pdstData[(i*width + j) * image.channels + c] = (Dtype)(pdata[(i*width + j) * 3 + c] - mean_value_[c]) * scale_;
				}
			}
		}

		return 0;
	}
	////后面考虑零拷贝，做内存池，直接指向该内存
	template <typename Dtype>
	int ImageDataLoader<Dtype>::readVec2Tensor(const std::vector<Dtype> &vec,
		std::shared_ptr<dlex_cnn::Tensor<Dtype>> datum, 
		const int num_idx)
	{
		//move to inline, delete judge?
		if (num_idx >= datum->getSize()[tind::eNum] ||
			image.channels != datum->getSize()[tind::eChannels] ||
			image.height != datum->getSize()[tind::eHeight] ||
			image.width != datum->getSize()[tind::eWidth])
		{
			//check 
			return -1;
		}
		if (normFlag == true && (scale_ <= 0 || mean_value_.size() != 1))
		{
			//check
			return -1;
		}

		const Dtype *pdstData = (Dtype *)datum->getData() + num_idx * datum->getSize()[tind::e3D];
		for (int i = 0; i < vec.size(); i++)
			pdstData[i] = vec[i];

		return 0;
	}
}
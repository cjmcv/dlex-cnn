////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Basic data structure
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "tensor.h"

namespace dlex_cnn
{
	template <typename Dtype>
	Tensor<Dtype>::Tensor(const int num, const int channels, const int height, const int width)
	{
		if (num <= 0 || channels <= 0 || height <= 0 || width <= 0)
		{
			DLOG_ERR("[ Tensor::Tensor ]: num <= 0 || channels <= 0 || height <= 0 || width <= 0.\n");
		}
		else if (num > MAX_SHAPE_SIZE || height > MAX_SHAPE_SIZE || width > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: num(%d) > MAX_SHAPE_SIZE || \
				height(%d) > MAX_SHAPE_SIZE || width(%d) > MAX_SHAPE_SIZE.\n", 
				num, height, width);
		}

		shape_.clear();
		shape_.push_back(num);
		shape_.push_back(channels);
		shape_.push_back(height);
		shape_.push_back(width);

		size_.clear();
		size_.push_back(width);
		size_.push_back(height * size_[0]);
		size_.push_back(channels * size_[1]);
		size_.push_back(num * size_[2]);
		
		data_ = NULL;
		data_ = (void *)malloc(sizeof(Dtype) * size_[size_.size() - 1]);

		if (data_ == NULL)
		{
			DLOG_ERR("[ Tensor::Tensor ]: Can not malloc for data_.\n");
		}
	}

	template <typename Dtype>
	Tensor<Dtype>::Tensor(const std::vector<int> &shape)
	{
		const int shapeSize = shape.size();
		if (shapeSize < 1 || shapeSize > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: shape.size() < 1 || shape.size() > MAX_SHAPE_SIZE.\n");
		}
		else if (shape[0] > MAX_SHAPE_SIZE || shape[2] > MAX_SHAPE_SIZE || shape[3] > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: num(%d) > MAX_SHAPE_SIZE || \
							height(%d) > MAX_SHAPE_SIZE || width(%d) > MAX_SHAPE_SIZE.\n",
							shape[0], shape[2], shape[3]);
		}

		shape_.clear();
		shape_ = shape;		
		
		size_.clear();
		size_.push_back(shape_[shapeSize - 1]);
		for (int i = 1; i < shapeSize; i++)
			size_.push_back(shape_[shapeSize - i - 1] * size_[i - 1]);
		
		data_ = NULL;
		data_ = (void *)malloc(sizeof(Dtype) * size_[tind::e4D]);
		if (data_ == NULL)
		{
			DLOG_ERR("[ Tensor::Tensor ]: Can not malloc for data_.\n");
		}
	}

	template <typename Dtype>
	Tensor<Dtype>::~Tensor()
	{
		if (data_ != NULL)
			free(data_);
		data_ = NULL;
	}

	template <typename Dtype>
	void Tensor<Dtype>::copyDataTo(Tensor<Dtype> &dstTensor)
	{
		if (dstTensor.shape_ != this->shape_ || dstTensor.size_ != this->size_)
		{
			DLOG_ERR("[ Tensor::copyDataTo ]: src tensor and dst tensor should have the same size.\n");
			return;
		}
		if (dstTensor.data_ == NULL || this->data_ == NULL)
		{
			DLOG_ERR("[ Tensor::copyDataTo ]: dstTensor.data_ == NULL || this->data_ == NULL.\n");
			return;
		}
		memcpy(dstTensor.data_, this->data_, sizeof(Dtype) * dstTensor.size_[tind::e4D]);
	}

	template <typename Dtype>
	void Tensor<Dtype>::cloneTo(Tensor<Dtype> &dstTensor)
	{
		dstTensor.shape_ = this->shape_;
		dstTensor.size_ = this->size_;

		if (dstTensor.data_ != NULL)
		{
			free(dstTensor.data_);
			dstTensor.data_ = NULL;
		}

		dstTensor.data_ = (void *)malloc(sizeof(Dtype) * this->size_[tind::e4D]);
		if (dstTensor.data_ == NULL)
			DLOG_ERR("[ Tensor::cloneTo ]: Can not malloc for dstTensor.data_.\n");

		memcpy(dstTensor.data_, this->data_, sizeof(Dtype) * dstTensor.size_[tind::e4D]);
	}

	INSTANTIATE_CLASS_NOR(Tensor);
}
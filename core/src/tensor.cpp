////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Basic data structure
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/device.h"
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
		
		cpu_data_ = NULL;
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
		
		cpu_data_ = NULL;
	}

	template <typename Dtype>
	void Tensor<Dtype>::checkCpuData()
	{
		if (cpu_data_ != NULL)	return;
		cpu_data_ = (void *)malloc(sizeof(Dtype) * size_[tind::e4D]);
		if (cpu_data_ == NULL)
		{
			DLOG_ERR("[ Tensor::Tensor ]: Can not malloc for cpu_data_.\n");
		}
	}

	template <typename Dtype>
	void Tensor<Dtype>::checkGpuData()
	{
		if (gpu_data_ != NULL)
			return;
#ifndef CPU_ONLY
		DCUDA_CHECK(cudaGetDevice(&gpu_device_));
		DCUDA_CHECK(cudaMalloc(&gpu_data_, sizeof(Dtype) * size_[tind::e4D]));
		DCUDA_CHECK(cudaMemset(gpu_data_, 0, sizeof(Dtype) * size_[tind::e4D]));	// needn't ?
#endif
		if (gpu_data_ == NULL)
		{
			DLOG_ERR("[ Tensor::Tensor ]: Can not malloc for cpu_data_.\n");
		}
	}

	template <typename Dtype>
	Tensor<Dtype>::~Tensor()
	{
		if (cpu_data_ != NULL)
			free(cpu_data_);
		cpu_data_ = NULL;
#ifndef CPU_ONLY
		if (gpu_data_ != NULL)
			cudaFree(gpu_data_);
		gpu_data_ = NULL;
#endif
	}

	template <typename Dtype>
	void Tensor<Dtype>::copyDataTo(Tensor<Dtype> &dst_tensor, tind::TensorCopyMode mode)
	{
		if (dst_tensor.shape_ != this->shape_ || dst_tensor.size_ != this->size_)
		{
			DLOG_ERR("[ Tensor::copyDataTo ]: src tensor and dst tensor should have the same size.\n");
			return;
		}
		switch(mode)
		{
		case tind::eHost2Host:
			if (dst_tensor.getCpuData() == NULL || this->cpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getCpuData() == NULL || this->cpu_data_ == NULL.\n");
				return;
			}
			memcpy(dst_tensor.cpu_data_, this->cpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D]);
			break;
#ifndef CPU_ONLY
		case tind::eHost2Device:
			if (dst_tensor.getGpuData() == NULL || this->cpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getGpuData() == NULL || this->cpu_data_ == NULL.\n");
				return;
			}
			DCUDA_CHECK(cudaMemcpy(dst_tensor.gpu_data_, this->cpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyHostToDevice));
			break;
		case tind::eDevice2Device:
			if (dst_tensor.getGpuData() == NULL || this->gpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getGpuData() == NULL || this->gpu_data_ == NULL.\n");
				return;
			}
			DCUDA_CHECK(cudaMemcpy(dst_tensor.gpu_data_, this->gpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyDeviceToDevice));
			break;
		case tind::eDevice2Host:
			if (dst_tensor.getCpuData() == NULL || this->gpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getCpuData() == NULL || this->gpu_data_ == NULL.\n");
				return;
			}
			DCUDA_CHECK(cudaMemcpy(dst_tensor.cpu_data_, this->gpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyDeviceToHost));
			break;
#endif
		default:
			DLOG_ERR("[ Tensor::copyDataTo ]: TensorCopyMode is invalid.\n");
			break;
		}
	}

	template <typename Dtype>
	void Tensor<Dtype>::cloneTo(Tensor<Dtype> &dst_tensor)
	{
		//if (this->cpu_data_ == NULL)
		//{
		//	DLOG_ERR("[ Tensor::cloneTo ]: this->cpu_data_ == NULL.\n");
		//	return;
		//}
		//dst_tensor.shape_ = this->shape_;
		//dst_tensor.size_ = this->size_;

		//if (dst_tensor.cpu_data_ != NULL)
		//{
		//	free(dst_tensor.cpu_data_);
		//	dst_tensor.cpu_data_ = NULL;
		//}

		//dst_tensor.cpu_data_ = (void *)malloc(sizeof(Dtype) * this->size_[tind::e4D]);
		//if (dst_tensor.cpu_data_ == NULL)
		//	DLOG_ERR("[ Tensor::cloneTo ]: Can not malloc for dst_tensor.cpu_data_.\n");

		//memcpy(dst_tensor.cpu_data_, this->cpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D]);
	}

	INSTANTIATE_CLASS_NOR(Tensor);
}
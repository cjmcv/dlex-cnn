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
			DLOG_ERR("[ Tensor::Tensor ]: num <= 0 || channels <= 0 || height <= 0 || width <= 0.");
		}
		else if (num > MAX_SHAPE_SIZE || height > MAX_SHAPE_SIZE || width > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: num(%d) > MAX_SHAPE_SIZE || \
					 					 height(%d) > MAX_SHAPE_SIZE || width(%d) > MAX_SHAPE_SIZE.",
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
		gpu_data_ = NULL;
		mem_head_ = tind::eUninitialized;
	}

	template <typename Dtype>
	Tensor<Dtype>::Tensor(const std::vector<int> &shape)
	{
		const int shapeSize = shape.size();
		if (shapeSize < 1 || shapeSize > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: shape.size() < 1 || shape.size() > MAX_SHAPE_SIZE.");
		}
		else if (shape[0] > MAX_SHAPE_SIZE || shape[2] > MAX_SHAPE_SIZE || shape[3] > MAX_SHAPE_SIZE)
		{
			DLOG_ERR("[ Tensor::Tensor ]: num(%d) > MAX_SHAPE_SIZE || \
					 					 height(%d) > MAX_SHAPE_SIZE || width(%d) > MAX_SHAPE_SIZE.",
										 shape[0], shape[2], shape[3]);
		}

		shape_.clear();
		shape_ = shape;

		size_.clear();
		size_.push_back(shape_[shapeSize - 1]);
		for (int i = 1; i < shapeSize; i++)
			size_.push_back(shape_[shapeSize - i - 1] * size_[i - 1]);

		cpu_data_ = NULL;
		gpu_data_ = NULL;
		mem_head_ = tind::eUninitialized;
	}

	template <typename Dtype>
	Tensor<Dtype>::~Tensor()
	{
		if (cpu_data_ != NULL)
			free(cpu_data_);
		cpu_data_ = NULL;
#ifdef USE_CUDA
		if (gpu_data_ != NULL)
			cudaFree(gpu_data_);
		gpu_data_ = NULL;
#endif
	}

	template <typename Dtype>
	int Tensor<Dtype>::mallocCpuData()
	{
		cpu_data_ = (void *)malloc(sizeof(Dtype) * size_[tind::e4D]);
		if (cpu_data_ == NULL)
		{
			DLOG_ERR("[ Tensor::mallocCpuData ]: Can not malloc for gpu_data_.");
			return -1;
		}
		else
			return 0;
	}
	template <typename Dtype>
	void Tensor<Dtype>::checkPushCpuData()
	{
		switch (mem_head_)
		{
		case tind::eUninitialized:
			mallocCpuData();
			set_cpu(size_[tind::e4D], (Dtype)0, (Dtype *)cpu_data_);
			mem_head_ = tind::eHeadAtCPU;
			break;
		case tind::eHeadAtGPU:
#ifdef USE_CUDA
			if (cpu_data_ == NULL)
				mallocCpuData();
			cpyInplace(tind::eDevice2Host);
			mem_head_ = tind::eHeadAtCPU;
#else
			DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif
			break;
		default:
			break;
		}
	}

#ifdef USE_CUDA
	template <typename Dtype>
	int Tensor<Dtype>::mallocGpuData()
	{
		CUDA_DCHECK(cudaGetDevice(&gpu_device_));
		CUDA_DCHECK(cudaMalloc(&gpu_data_, sizeof(Dtype) * size_[tind::e4D]));
		if (gpu_data_ == NULL)
		{
			DLOG_ERR("[ Tensor::mallocGpuData ]: Can not malloc for gpu_data_.");
			return -1;
		}
		else
			return 0;
	}
	template <typename Dtype>
	void Tensor<Dtype>::checkPushGpuData()
	{
		switch (mem_head_)
		{
		case tind::eUninitialized:
			mallocGpuData();
			CUDA_DCHECK(cudaMemset(gpu_data_, 0, sizeof(Dtype) * size_[tind::e4D]));
			mem_head_ = tind::eHeadAtGPU;
			break;
		case tind::eHeadAtCPU:
			if (gpu_data_ == NULL)
				mallocGpuData();
			cpyInplace(tind::eHost2Device);
			mem_head_ = tind::eHeadAtGPU;
			break;
		default:
			break;
		}
	}
#endif

	template <typename Dtype>
	void Tensor<Dtype>::copyDataTo(Tensor<Dtype> &dst_tensor, tind::TensorCopyMode cp_mode)
	{
		if (dst_tensor.shape_ != this->shape_ || dst_tensor.size_ != this->size_)
		{
			DLOG_ERR("[ Tensor::copyDataTo ]: src tensor and dst tensor should have the same size.");
			return;
		}
		switch (cp_mode)
		{
		case tind::eHost2Host:
			if (dst_tensor.getCpuData() == NULL || this->cpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getCpuData() == NULL || this->cpu_data_ == NULL.");
				return;
			}

			memcpy(dst_tensor.cpu_data_, this->cpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D]);
			dst_tensor.setMemHead(tind::eHeadAtCPU);
			break;
#ifdef USE_CUDA
		case tind::eHost2Device:
			if (dst_tensor.getGpuData() == NULL || this->cpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getGpuData() == NULL || this->cpu_data_ == NULL.");
				return;
			}
			CUDA_DCHECK(cudaMemcpy(dst_tensor.gpu_data_, this->cpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyHostToDevice));
			dst_tensor.setMemHead(tind::eHeadAtGPU);
			break;
		case tind::eDevice2Device:
			if (dst_tensor.getGpuData() == NULL || this->gpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getGpuData() == NULL || this->gpu_data_ == NULL.");
				return;
			}
			CUDA_DCHECK(cudaMemcpy(dst_tensor.gpu_data_, this->gpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyDeviceToDevice));
			dst_tensor.setMemHead(tind::eHeadAtGPU);
			break;
		case tind::eDevice2Host:
			if (dst_tensor.getCpuData() == NULL || this->gpu_data_ == NULL)
			{
				DLOG_ERR("[ Tensor::copyDataTo ]: dst_tensor.getCpuData() == NULL || this->gpu_data_ == NULL.");
				return;
			}
			CUDA_DCHECK(cudaMemcpy(dst_tensor.cpu_data_, this->gpu_data_, sizeof(Dtype) * dst_tensor.size_[tind::e4D], cudaMemcpyDeviceToHost));
			dst_tensor.setMemHead(tind::eHeadAtCPU);
			break;
#endif
		default:
			DLOG_ERR("[ Tensor::copyDataTo ]: TensorCopyMode is invalid.");
			break;
		}
	}

#ifdef USE_CUDA
	template <typename Dtype>
	void Tensor<Dtype>::asyncCpy2GPU(const cudaStream_t& stream)
	{
		if (!DCHECK(this->cpu_data_ != NULL))
			DLOG_ERR("this->cpu_data_ == NULL");

		CUDA_DCHECK(cudaMemcpyAsync(getPushGpuData(), cpu_data_, sizeof(Dtype) * size_[tind::e4D], cudaMemcpyHostToDevice, stream));
		mem_head_ = tind::eSynced;
	}
	template <typename Dtype>
	void Tensor<Dtype>::cpyInplace(tind::TensorCopyMode cp_mode)
	{
		switch (cp_mode)
		{
		case tind::eHost2Device:
			if (cpu_data_ == NULL || gpu_data_ == NULL)
				DLOG_ERR("cpu_data_ == NULL || gpu_data_ == NULL");
			CUDA_DCHECK(cudaMemcpy(gpu_data_, cpu_data_, sizeof(Dtype) * size_[tind::e4D], cudaMemcpyHostToDevice));
			break;
		case tind::eDevice2Host:
			if (cpu_data_ == NULL || gpu_data_ == NULL)
				DLOG_ERR("cpu_data_ == NULL || gpu_data_ == NULL");
			CUDA_DCHECK(cudaMemcpy(cpu_data_, gpu_data_, sizeof(Dtype) * size_[tind::e4D], cudaMemcpyDeviceToHost));
			break;
		default:
			DLOG_ERR("Invalid copy mode %d", cp_mode);
			break;
		}
	}
#endif

	INSTANTIATE_CLASS_NOR(Tensor);
}

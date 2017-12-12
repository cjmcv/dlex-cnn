////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Basic data structure
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TENSOR_HPP_
#define DLEX_TENSOR_HPP_

//#include <iostream>
#include <vector>
#include <string.h>
#include <memory>
#include <stdlib.h>

#include "common.h"
#include "util/math_functions.h"
#include "util/device.h"

#define MAX_SHAPE_SIZE 9999

namespace dlex_cnn
{
	namespace tind
	{
		enum TensorSizeIndex  { e1D, e2D, e3D, e4D };
		enum TensorShapeIndex { eNum, eChannels, eHeight, eWidth };
		enum TensorCopyMode { eHost2Host, eHost2Device, eDevice2Device, eDevice2Host };
		enum TensorMemHead { eUninitialized, eHeadAtCPU, eHeadAtGPU, eSynced };
	}

	template <typename Dtype>
	class Tensor
	{
	public:
		explicit Tensor() { cpu_data_ = NULL; gpu_data_ = NULL; mem_head_ = tind::eUninitialized; };
		explicit Tensor(const int num, const int channels, const int height, const int width);
		explicit Tensor(const std::vector<int> &shape);
		virtual ~Tensor();
		
		inline std::vector<int> &getSize() { return size_; };
		inline std::vector<int> &getShape() { return shape_; };	
		
		inline void setMemHead(tind::TensorMemHead mem_head) { mem_head_ = mem_head; }

		int mallocCpuData();
		inline void *getCpuData() {
			if (cpu_data_ == NULL)
				mallocCpuData();
			return cpu_data_;
		}
		void checkPushCpuData();
		inline void *getPushCpuData() {
			checkPushCpuData();
			return cpu_data_; 
		}

		inline void setCpuZero() { set_cpu(size_[tind::e4D], (Dtype)0, (Dtype *)getCpuData()); };
		inline void setCpuValue(Dtype alpha) {
			Dtype *dst = (Dtype *)getCpuData();
			set_cpu(size_[tind::e4D], alpha, dst);
		};

		// Just copy data, without changing their size
		void copyDataTo(Tensor<Dtype> &dst_tensor, tind::TensorCopyMode mode);

#ifdef USE_CUDA
		int mallocGpuData();
		inline void *getGpuData() {
			if (gpu_data_ == NULL)
				mallocGpuData();
			return gpu_data_;
		}
		void checkPushGpuData();
		inline void *getPushGpuData() {
			checkPushGpuData();
			return gpu_data_;
		}

		inline void setGpuZero() { set_gpu(size_[tind::e4D], (Dtype)0, (Dtype *)getGpuData()); };
		inline void setGpuValue(Dtype alpha) {
			Dtype *dst = (Dtype *)getGpuData();
			set_gpu(size_[tind::e4D], alpha, dst);
		};

		// Push data from cpu to gpu.
		void asyncCpy2GPU(const cudaStream_t& stream);
		
		void cpyInplace(tind::TensorCopyMode cp_mode);
#endif
	private:
		int mem_head_;
		void *cpu_data_;
		void *gpu_data_;
		int gpu_device_;

		// eNum, eChannels, eHeight, eWidth
		std::vector<int> shape_;

		// size_[0] = width, 
		// size_[1] = height * width,
		// size_[2] = channels * height * width
		// size_[3] = num * channels * height * width;
		std::vector<int> size_;
	};

}


#endif //DLEX_TENSOR_HPP_

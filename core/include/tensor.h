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
#include "util/device.h"

#define MAX_SHAPE_SIZE 9999

namespace dlex_cnn
{
	namespace tind
	{
		enum TensorSizeIndex  { e1D, e2D, e3D, e4D };
		enum TensorShapeIndex { eNum, eChannels, eHeight, eWidth };
		enum TensorCopyMode { eHost2Host, eHost2Device, eDevice2Device, eDevice2Host };
	}

	template <typename Dtype>
	class Tensor
	{
	public:
		explicit Tensor() { cpu_data_ = NULL; };
		explicit Tensor(const int num, const int channels, const int height, const int width);
		explicit Tensor(const std::vector<int> &shape);
		virtual ~Tensor();
		
		void checkCpuData();
		void checkGpuData();
		inline std::vector<int> &getSize() { return size_; };
		inline std::vector<int> &getShape() { return shape_; };
		inline void *getCpuData() {
			checkCpuData();
			return cpu_data_; 
		}
		inline void *getGpuData() {
			checkGpuData();
			return gpu_data_;
		}
		inline void setZero() { memset(getCpuData(), 0, sizeof(Dtype) * size_[tind::e4D]); };
		inline void setValue(Dtype alpha) {
			Dtype *dst = (Dtype *)getCpuData();
			for (int i = 0; i < size_[tind::e4D]; ++i) {
				dst[i] = alpha;
			}
		};
		// Just copy data, without changing their size
		void copyDataTo(Tensor<Dtype> &dst_tensor, tind::TensorCopyMode mode);
		// Push data from cpu to gpu.
		void asyncCpy2GPU(const cudaStream_t& stream);

	private:
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

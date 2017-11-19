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
#include <memory>
#include <stdlib.h>

#include "common.h"

#define MAX_SHAPE_SIZE 9999

namespace dlex_cnn
{
	namespace tind
	{
		enum TensorSizeIndex  { e1D, e2D, e3D, e4D };
		enum TensorShapeIndex { eNum, eChannels, eHeight, eWidth };
	}

	template <typename Dtype>
	class Tensor
	{
	public:
		explicit Tensor() { data_ = NULL; };
		explicit Tensor(const int num, const int channels, const int height, const int width);
		explicit Tensor(const std::vector<int> &shape);
		virtual ~Tensor();
		
		inline std::vector<int> &getSize() { return size_; };
		inline std::vector<int> &getShape() { return shape_; };
		inline void *getData() { return data_; };
		inline void setZero() { memset(data_, 0, sizeof(Dtype) * size_[tind::e4D]); };
		inline void setValue(Dtype alpha) {
			Dtype *dst = (Dtype *)data_;
			for (int i = 0; i < size_[tind::e4D]; ++i) {
				dst[i] = alpha;
			}
		};
		// Just copy data, without changing their size
		void copyDataTo(Tensor<Dtype> &dstTensor);
		// Copy the whole tensor, includes their size and data.
		void cloneTo(Tensor<Dtype> &dstTensor);

	private:
		void *data_;
		
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
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_MNIST_LOADER_HPP_
#define DLEX_TEST_MNIST_LOADER_HPP_

#include <vector>
#include <cstdint>

#include "io/dataloader.h"

namespace dlex_cnn
{
	class MnistLoader
	{
	public:
		MnistLoader() {};
		virtual ~MnistLoader() {};

	private:
		template<typename T>
		inline T reverse_endian(T p) {
			std::reverse(reinterpret_cast<char*>(&p), reinterpret_cast<char*>(&p) + sizeof(T));
			return p;
		}

		inline bool is_little_endian() {
			int x = 1;
			return *(char*)&x != 0;
		}

	public:
		bool load_mnist_images(const std::string& file_path, std::vector<dlex_cnn::IMAGE_DATUM>& images);
		bool load_mnist_labels(const std::string& file_path, std::vector<char>& labels);
	};
}

#endif
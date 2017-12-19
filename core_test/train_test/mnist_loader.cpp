////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Load mnist data.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <fstream>
#include <cassert>
#include "mnist_loader.h"

namespace dlex_cnn
{
	// Notice the memory release of IMAGE_DATUM.
	bool MnistLoader::load_mnist_images(const std::string& file_path, std::vector<dlex_cnn::IMAGE_DATUM>& images)
	{
		images.clear();
		std::ifstream ifs(file_path, std::ios::binary);
		if (!ifs.is_open())
		{
			return false;
		}
		//detect platform information
		const bool is_little_endian_flag = is_little_endian();
		//magic number
		uint32_t magic_number;
		ifs.read((char*)&magic_number, sizeof(magic_number));
		if (is_little_endian_flag)
		{
			magic_number = reverse_endian<uint32_t>(magic_number);
		}
		const bool magic_number_validate = (magic_number == 0x00000803);
		if (!magic_number_validate)
			return false;

		//count
		uint32_t images_total_count = 0;
		ifs.read((char*)&images_total_count, sizeof(images_total_count));
		//image property
		uint32_t width = 0, height = 0;
		ifs.read((char*)&height, sizeof(height));
		ifs.read((char*)&width, sizeof(width));
		if (is_little_endian_flag)
		{
			images_total_count = reverse_endian<uint32_t>(images_total_count);
			width = reverse_endian<uint32_t>(width);
			height = reverse_endian<uint32_t>(height);
		}
		//images
		for (uint32_t i = 0; i < images_total_count; i++)
		{
			dlex_cnn::IMAGE_DATUM image;
			image.channels = 1;
			image.width = width;
			image.height = height;
			//image.data.resize(width*height);
			image.pdata = new unsigned char[width*height];
			ifs.read((char*)image.pdata, width*height);
			images.push_back(image);
		}
		return true;
	}

	bool MnistLoader::load_mnist_labels(const std::string& file_path, std::vector<char>& labels)
	{
		labels.clear();
		std::ifstream ifs(file_path, std::ios::binary);
		if (!ifs.is_open())
		{
			return false;
		}//detect platform information
		const bool is_little_endian_flag = is_little_endian();
		//magic number
		uint32_t magic_number;
		ifs.read((char*)&magic_number, sizeof(magic_number));
		if (is_little_endian_flag)
		{
			magic_number = reverse_endian<uint32_t>(magic_number);
		}
		const bool magic_number_validate = (magic_number == 0x00000801);
		if (!magic_number_validate)
			return false;
		//count
		uint32_t labels_total_count = 0;
		ifs.read((char*)&labels_total_count, sizeof(labels_total_count));
		if (is_little_endian_flag)
		{
			labels_total_count = reverse_endian<uint32_t>(labels_total_count);
		}
		//labels
		for (uint32_t i = 0; i < labels_total_count; i++)
		{
			char label;
			ifs.read((char*)&label, sizeof(char));
			labels.push_back(label);
		}
		return true;
	}
}
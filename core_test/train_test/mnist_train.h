////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_MNIST_TRAIN_HPP_
#define DLEX_TEST_MNIST_TRAIN_HPP_
//
#include <iostream>
#include <cassert>
#include "dlex_cnn.h"
#include "mnist_loader.h"
#include "trainer/network_creator.h"

namespace dlex_cnn
{
	class MnistTrainTest
	{
	public:
		MnistTrainTest();
		virtual ~MnistTrainTest() {};
	
	public:		
		void train();
		void test(const std::string& model_file_path, const int iter);

	private:
		bool loadMnistData(tind::Phase phase);
		bool releaseMnistData();
		bool fetchBatchData(const std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> &train_data,
			std::shared_ptr<dlex_cnn::Tensor<float>> input_data_tensor,
			std::shared_ptr<dlex_cnn::Tensor<float>> label_data_tensor,
			const int offset, const int length);

		std::shared_ptr<dlex_cnn::Tensor<float>> convertLabelToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len);
		std::shared_ptr<dlex_cnn::Tensor<float>> convertVectorToTensor(const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data, const int start, const int len);
		uint8_t getMaxIdxInArray(const float* start, const float* stop);
		std::pair<float, float> testInTrain(dlex_cnn::NetWork<float>& network, const int batch, const std::vector< std::pair<dlex_cnn::IMAGE_DATUM, char> > &test_data);

	private:
		int class_num_;

		std::string train_images_file_;
		std::string train_labels_file_;

		std::string test_images_file_;
		std::string test_labels_file_;

		std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> train_data_;
		std::vector<std::pair<dlex_cnn::IMAGE_DATUM, char>> validate_data_;
	};
}

void mnistTrain();
void mnistTest();

#endif

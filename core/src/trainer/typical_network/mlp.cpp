////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "trainer/typical_network.h"

namespace dlex_cnn
{
	template <typename Dtype>
	int TypicalNet<Dtype>::mlp(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network)
	{
		printf("start building net\n");
		NetCreator<Dtype> creator;

		InputOpParam input_params;
		input_params.num = num;
		input_params.channels = channels;
		input_params.height = height;
		input_params.width = width;
		std::string input_name = "input";
		creator.createInputNode(input_name, input_params, network);

		std::string fc0_params = "blas_enable:1, num_hidden:256,";
		std::string fc0_name = "fc0";
		creator.createInnerProductNode(input_name, fc0_name, fc0_params, network);

		std::string fc1_params = "blas_enable:1, num_hidden:256,";
		std::string fc1_name = "fc1";
		creator.createInnerProductNode(fc0_name, fc1_name, fc1_params, network);

		std::string fc2_params = "blas_enable:1, num_hidden:10,";
		std::string fc2_name = "fc2";
		creator.createInnerProductNode(fc1_name, fc2_name, fc2_params, network);

		std::string sm_params = ",";
		std::string sm_name = "sm";
		creator.createSoftmaxLossNode(fc2_name, sm_name, sm_params, network);

		std::string out_params = "label_dim:1,";
		std::string out_name = "output";
		creator.createOutputNode(sm_name, out_name, out_params, network);

		creator.createOptimizer("SGD", network);

		return 0;
	}

	
}
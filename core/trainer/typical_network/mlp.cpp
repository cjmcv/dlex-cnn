#include "trainer/typical_network.h"

namespace dlex_cnn
{
	template <typename Dtype>
	int TypicalNet<Dtype>::mlp(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network)
	{
		printf("start building net\n");
		NetCreator<Dtype> netBase;

		InputOpParam input_params;
		input_params.num = num;
		input_params.channels = channels;
		input_params.height = height;
		input_params.width = width;
		std::string input_name = "input";
		netBase.createInputNode(input_name, input_params, network);

		std::string fc0_params = "blas_enable:1, num_hidden:256,";
		std::string fc0_name = "fc0";
		netBase.createInnerProductNode(input_name, fc0_name, fc0_params, network);

		std::string fc1_params = "blas_enable:1, num_hidden:256,";
		std::string fc1_name = "fc1";
		netBase.createInnerProductNode(fc0_name, fc1_name, fc1_params, network);

		std::string fc2_params = "blas_enable:1, num_hidden:10,";
		std::string fc2_name = "fc2";
		netBase.createInnerProductNode(fc1_name, fc2_name, fc2_params, network);

		std::string sm_params = ",";
		std::string sm_name = "sm";
		netBase.createSoftmaxLossNode(fc2_name, sm_name, sm_params, network);

		std::string out_params = "label_dim:1,";
		std::string out_name = "output";
		netBase.createOutputNode(sm_name, out_name, out_params, network);

		return 0;
	}

	
}
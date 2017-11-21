#include "trainer/typical_network.h"

namespace dlex_cnn
{
	template <typename Dtype>
	int TypicalNet<Dtype>::lenet(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network)
	{
		printf("start building net\n");
		NetCreator<Dtype> netCreator;

		InputOpParam input_params;
		input_params.num = num;
		input_params.channels = channels;
		input_params.height = height;
		input_params.width = width;
		std::string input_name = "input";
		netCreator.createInputNode(input_name, input_params, network);

		std::string conv1_params = "blas_enable:1, kernel_num:20,  kernel_h:5, kernel_w:5, stride_h:1, stride_w:1,  pad_h:0, pad_w:0, dilation_h:1, dilation_w:1,";
		std::string conv1_name = "conv1";
		netCreator.createConvNode(input_name, conv1_name, conv1_params, network);

		std::string pool1_params = "poolingType:0,kernel_h:2,kernel_w:2,stride_h:2,stride_w:2,pad_h:0,pad_w:0,global_pooling:0,";
		std::string pool1_name = "pool1";
		netCreator.createPoolNode(conv1_name, pool1_name, pool1_params, network);

		std::string conv2_params = "blas_enable:1, kernel_num:50,  kernel_h:5, kernel_w:5, stride_h:1, stride_w:1,  pad_h:0, pad_w:0, dilation_h:1, dilation_w:1,";
		std::string conv2_name = "conv2";
		netCreator.createConvNode(pool1_name, conv2_name, conv2_params, network);

		std::string pool2_params = "poolingType:0,kernel_h:2,kernel_w:2,stride_h:2,stride_w:2,pad_h:0,pad_w:0,global_pooling:0,";
		std::string pool2_name = "pool2";
		netCreator.createPoolNode(conv2_name, pool2_name, pool2_params, network);

		std::string fc1_params = "blas_enable:1, num_hidden:500,";
		std::string fc1_name = "fc1";
		netCreator.createInnerProductNode(pool2_name, fc1_name, fc1_params, network);

		std::string act1_params = "activationType:0, negative_slope:0, ";
		std::string act1_name = "act1";
		netCreator.createActivationNode(fc1_name, act1_name, act1_params, network);

		std::string fc2_params = "blas_enable:1, num_hidden:10,";
		std::string fc2_name = "fc2";
		netCreator.createInnerProductNode(act1_name, fc2_name, fc2_params, network);

		std::string sm_params = ",";
		std::string sm_name = "sm";
		netCreator.createSoftmaxLossNode(fc2_name, sm_name, sm_params, network);

		std::string out_params = "label_dim:1,";
		std::string out_name = "output";
		netCreator.createOutputNode(sm_name, out_name, out_params, network);

		return 0;
	}


}
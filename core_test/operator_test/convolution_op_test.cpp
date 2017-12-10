////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_cnn.h"
#include "common/tools.h"
#include "convolution_op_test.h"

//#include "../core/operator/convolution_op.h"

#ifdef USE_OP_TEST
namespace dlex_cnn {

	template <typename Dtype>
	void ConvolutionOpTest<Dtype>::forward()
	{
		registerOpClass();

		std::shared_ptr<dlex_cnn::Op<float>> conv1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Convolution");
		dlex_cnn::ConvolutionOpParam conv_param;
		conv_param.blas_enable = true;
		conv_param.kernel_num = 2;
		conv_param.kernel_h = 2;
		conv_param.kernel_w = 2;
		conv_param.pad_h = 0;
		conv_param.pad_w = 0;
		conv_param.stride_h = 1;
		conv_param.stride_w = 1;
		conv_param.dilation_h = 1;
		conv_param.dilation_w = 1;

		dlex_cnn::ConvolutionOp<float>* conv1 = dynamic_cast<dlex_cnn::ConvolutionOp<float> *>(conv1_s.get());
		conv1->setOpParam(conv_param);

		int is[4] = {1,3,3,3};
		std::vector<int> in_shape;
		for (int i = 0; i < 4; i++)
			in_shape.push_back(is[i]);

		std::vector<int> out_shape;
		conv1->inferOutShape(in_shape, out_shape);

		std::vector<std::shared_ptr<Tensor<float>>> in_data_vec;
		conv1->allocBuf4Node(in_shape, out_shape, in_data_vec);

		normal_distribution_init<float>(in_data_vec[1]->getSize()[tind::e4D], 0.0f, 0.1f, (float *)in_data_vec[1]->getCpuData());
		if (conv1->param_.blas_enable)
			dlex_set<float>(in_data_vec[2]->getSize()[tind::e4D], 0.0f, (float *)in_data_vec[2]->getCpuData());

		// input (ic3, ih3, iw3)
		float in_buffer[] = {1,2,0,1,1,3,0,2,2, 0,2,1,0,3,2,1,1,0, 1,2,1,0,1,3,3,3,2};
		float *in_data = (float *)in_data_vec[0]->getCpuData();
		for (int i = 0; i < 1*3*3*3; i++)
			in_data[i] = in_buffer[i];
		// weight (kn2, kc3 = ic3, kh2, kw2) 
		float w_buffer[] = {1,1,2,2, 1,1,1,1, 0,1,1,0, 1,0,0,1, 2,1,2,1, 1,2,2,0};
		float *w_data = (float *)in_data_vec[1]->getCpuData();
		for (int i = 0; i < 2*3*2*2; i++)
			w_data[i] = w_buffer[i];
		// bias ()
		//float b_buffer[] = { 1, 2, 0, 1, 1, 3, 0, 2, 2, 0, 2, 1, 0, 3, 2, 1, 1, 0, 1, 2, 1, 0, 1, 3, 3, 3, 2 };
		//float *in_data = (float *)in_data_vec[0]->getCpuData();
		//for (int i = 0; i < in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3]; i++)
		//	in_data[i] = in_buffer[i];

		conv1->allocOpBuf4Train(in_shape, out_shape);

		std::vector<std::shared_ptr<Tensor<float>>> out_data_vec;
		out_data_vec.push_back(std::make_shared<Tensor<float>>(out_shape));

		conv1->forward(in_data_vec, out_data_vec);

		//matrixShow_float("A", (float *)in_data_vec[0]->getCpuData(), in_data_vec[0]->getShape()[tind::eNum], in_data_vec[0]->getShape()[tind::eChannels], in_data_vec[0]->getShape()[tind::eHeight], in_data_vec[0]->getShape()[tind::eWidth]);
		//matrixShow_float("B", (float *)out_data_vec[0]->getCpuData(), out_data_vec[0]->getShape()[tind::eNum], out_data_vec[0]->getShape()[tind::eChannels], out_data_vec[0]->getShape()[tind::eHeight], out_data_vec[0]->getShape()[tind::eWidth]);

		//反向传播，对比，矩阵手动计算对比
		std::vector<std::shared_ptr<Tensor<Dtype>>> in_diff;
		in_diff.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
		std::vector<std::shared_ptr<Tensor<Dtype>>> out_diff;
		out_diff.push_back(std::make_shared<Tensor<Dtype>>(out_shape));

		// set out here first.
		conv1->backward(in_data_vec, out_data_vec, in_diff, out_data_vec);
		matrixShow_float("C", (float *)in_diff[0]->getCpuData(), in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);

		const std::vector<int> kernel_shape = in_data_vec[1]->getShape();
		matrixShow_float("weight gradient", (float *)conv1->gradient_[0]->getCpuData(), kernel_shape[tind::eNum], kernel_shape[tind::eChannels], kernel_shape[tind::eHeight], kernel_shape[tind::eWidth]);

		const std::vector<int> bias_shape = in_data_vec[2]->getShape();
		matrixShow_float("bias gradient", (float *)conv1->gradient_[1]->getCpuData(), bias_shape[tind::eNum], bias_shape[tind::eChannels], bias_shape[tind::eHeight], bias_shape[tind::eWidth]);
	}

}

void testConv()
{
	dlex_cnn::ConvolutionOpTest<float> conv_test;
	conv_test.forward();
}
#endif

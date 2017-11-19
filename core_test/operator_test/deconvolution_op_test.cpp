////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_cnn.h"
#include "common/tools.h"
#include "deconvolution_op_test.h"

//#include "../core/operator/convolution_op.h"

#ifdef UNIT_TEST
namespace dlex_cnn {

	template <typename Dtype>
	void DeconvolutionOpTest<Dtype>::forward()
	{
		registerOpClass();

		std::shared_ptr<dlex_cnn::Op<float>> conv1_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Deconvolution");
		dlex_cnn::DeconvolutionOpParam DeconvolutionParam;
		DeconvolutionParam.blas_enable = true;
		DeconvolutionParam.kernel_channels = 3;
		DeconvolutionParam.kernel_h = 2;
		DeconvolutionParam.kernel_w = 2;
		DeconvolutionParam.pad_h = 0;
		DeconvolutionParam.pad_w = 0;
		DeconvolutionParam.stride_h = 1;
		DeconvolutionParam.stride_w = 1;
		DeconvolutionParam.dilation_h = 1;
		DeconvolutionParam.dilation_w = 1;

		dlex_cnn::DeconvolutionOp<float>* conv1 = dynamic_cast<dlex_cnn::DeconvolutionOp<float> *>(conv1_s.get());
		conv1->setOpParam(DeconvolutionParam);

		int is[4] = {1,2,3,3};	//3
		std::vector<int> inShape;
		for (int i = 0; i < 4; i++)
			inShape.push_back(is[i]);

		std::vector<int> outShape;
		conv1->inferOutShape(inShape, outShape);

		std::vector<std::shared_ptr<Tensor<float>>> inDataVec;
		conv1->allocBuf4Node(inShape, outShape, inDataVec);

		// input (ic2, ih3, iw3)
		float in_buffer[] = {1,2,0,1,1,3,0,2,2, 0,2,1,0,3,2,1,1,0}; //
		//float in_buffer[] = {1,0,0,0, 0,1,0,0, 0,0,3,0, 0,0,0,1};
		float *inData = (float *)inDataVec[0]->getData();
		for (int i = 0; i < 1*2*3*3; i++)
			inData[i] = in_buffer[i];
		// weight (kn2 = ic2, kc3, kh2, kw2) 
		float w_buffer[] = {1,1,2,2, 1,1,1,1, 0,1,1,3,  1,0,0,1, 2,1,2,1, 1,2,2,0};	//2*3*2*2
		//float w_buffer[] = {4,5,3,4};	//2*2*2*2
		float *wData = (float *)inDataVec[1]->getData();
		for (int i = 0; i < 2*3*2*2; i++)
			wData[i] = w_buffer[i];
		// bias ()
		//float b_buffer[] = { 1, 2, 0, 1, 1, 3, 0, 2, 2, 0, 2, 1, 0, 3, 2, 1, 1, 0, 1, 2, 1, 0, 1, 3, 3, 3, 2 };
		//float *inData = (float *)inDataVec[0]->getData();
		//for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; i++)
		//	inData[i] = in_buffer[i];

		conv1->allocOpBuf4Train(inShape, outShape);

		std::vector<std::shared_ptr<Tensor<float>>> outDataVec;
		outDataVec.push_back(std::make_shared<Tensor<float>>(outShape));

		conv1->forward(inDataVec, outDataVec);

		matrixShow_float("A", (float *)inDataVec[0]->getData(), 
			inDataVec[0]->getShape()[tind::eNum], 
			inDataVec[0]->getShape()[tind::eChannels], 
			inDataVec[0]->getShape()[tind::eHeight],
			inDataVec[0]->getShape()[tind::eWidth]);
		matrixShow_float("W", (float *)inDataVec[1]->getData(),
			inDataVec[1]->getShape()[tind::eNum],
			inDataVec[1]->getShape()[tind::eChannels],
			inDataVec[1]->getShape()[tind::eHeight],
			inDataVec[1]->getShape()[tind::eWidth]);
		matrixShow_float("B", (float *)outDataVec[0]->getData(), 
			outDataVec[0]->getShape()[tind::eNum], 
			outDataVec[0]->getShape()[tind::eChannels], 
			outDataVec[0]->getShape()[tind::eHeight], 
			outDataVec[0]->getShape()[tind::eWidth]);

		//反向传播，对比，矩阵手动计算对比
		std::vector<std::shared_ptr<Tensor<Dtype>>> inDiff;
		inDiff.push_back(std::make_shared<Tensor<Dtype>>(inShape));
		std::vector<std::shared_ptr<Tensor<Dtype>>> outDiff;
		outDiff.push_back(std::make_shared<Tensor<Dtype>>(outShape));

		// set out here first.
		conv1->backward(inDataVec, outDataVec, inDiff, outDataVec);
		matrixShow_float("C", (float *)inDiff[0]->getData(), 
			inShape[tind::eNum], inShape[tind::eChannels], 
			inShape[tind::eHeight], inShape[tind::eWidth]);

		const std::vector<int> kernelShape = inDataVec[1]->getShape();
		matrixShow_float("weight gradient", (float *)conv1->gradient_[0]->getData(), 
			kernelShape[tind::eNum], kernelShape[tind::eChannels], 
			kernelShape[tind::eHeight], kernelShape[tind::eWidth]);

		const std::vector<int> biasShape = inDataVec[2]->getShape();
		matrixShow_float("bias gradient", (float *)conv1->gradient_[1]->getData(),
			biasShape[tind::eNum], biasShape[tind::eChannels], 
			biasShape[tind::eHeight], biasShape[tind::eWidth]);
	}

}

void testDeconv()
{
	dlex_cnn::DeconvolutionOpTest<float> convTest;
	convTest.forward();
	system("pause");
}
#endif
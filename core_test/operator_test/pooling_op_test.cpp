////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_cnn.h"
#include "common/tools.h"
#include "pooling_op_test.h"

#ifdef UNIT_TEST
namespace dlex_cnn {

	template <typename Dtype>
	void PoolingOpTest<Dtype>::forward()
	{
		registerOpClass();

		std::shared_ptr<dlex_cnn::Op<float>> pool_s = dlex_cnn::OpFactory<float>::getInstance().createOpByType("Pooling");
		dlex_cnn::PoolingOpParam PoolingParam;
		PoolingParam.global_pooling = true;
		PoolingParam.kernel_h = 3;
		PoolingParam.kernel_w = 3;
		PoolingParam.pad_h = 0;
		PoolingParam.pad_w = 0;
		PoolingParam.stride_h = 3;
		PoolingParam.stride_w = 3;
		PoolingParam.poolingType = dlex_cnn::tind::eAVE;
		dlex_cnn::PoolingOp<float>* pool = dynamic_cast<dlex_cnn::PoolingOp<float> *>(pool_s.get());
		pool->setOpParam(PoolingParam);

		int is[4] = { 1, 1, 10, 10 };
		std::vector<int> inShape;
		for (int i = 0; i < 4; i++)
			inShape.push_back(is[i]);

		std::vector<int> outShape;
		pool->inferOutShape(inShape, outShape);

		std::vector<std::shared_ptr<Tensor<float>>> inDataVec;
		pool->allocBuf4Node(inShape, outShape, inDataVec);

		pool->allocOpBuf4Train(inShape, outShape);

		// input (ic3, ih3, iw3)
		float *inData = (float *)inDataVec[0]->getData();
		for (int i = 0; i < 1 * 1 * 10 * 10; i++)
			inData[i] = i;

		pool->allocOpBuf4Train(inShape, outShape);

		std::vector<std::shared_ptr<Tensor<float>>> outDataVec;
		outDataVec.push_back(std::make_shared<Tensor<float>>(outShape));

		pool->forward(inDataVec, outDataVec);

		matrixShow_float("A", (float *)inDataVec[0]->getData(), inDataVec[0]->getShape()[tind::eNum], inDataVec[0]->getShape()[tind::eChannels], inDataVec[0]->getShape()[tind::eHeight], inDataVec[0]->getShape()[tind::eWidth]);
		matrixShow_float("B", (float *)outDataVec[0]->getData(), outDataVec[0]->getShape()[tind::eNum], outDataVec[0]->getShape()[tind::eChannels], outDataVec[0]->getShape()[tind::eHeight], outDataVec[0]->getShape()[tind::eWidth]);
		if (PoolingParam.poolingType = dlex_cnn::tind::eMAX)
			matrixShow_int("C", (int *)pool->max_idx_map_->getData(), outDataVec[0]->getShape()[tind::eNum], outDataVec[0]->getShape()[tind::eChannels], outDataVec[0]->getShape()[tind::eHeight], outDataVec[0]->getShape()[tind::eWidth]);

		//反向传播，对比，矩阵手动计算对比
		std::vector<std::shared_ptr<Tensor<Dtype>>> inDiffVec;
		inDiffVec.push_back(std::make_shared<Tensor<Dtype>>(inShape));
		std::vector<std::shared_ptr<Tensor<Dtype>>> outDiff;
		outDiff.push_back(std::make_shared<Tensor<Dtype>>(outShape));

		// set out here first.
		pool->backward(inDataVec, outDataVec, inDiffVec, outDataVec);
		matrixShow_float("D", (float *)inDiffVec[0]->getData(), inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
	}

}

void testPool()
{
	dlex_cnn::PoolingOpTest<float> poolTest;
	poolTest.forward();
}
#endif
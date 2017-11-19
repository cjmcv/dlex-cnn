////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/cross_entropy_lop.h"
#include "util/math_functions.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	CrossEntropyLossOp<Dtype>::CrossEntropyLossOp()
	{
		op_type_ = "CrossEntropyLoss";
		labels_ = NULL;
	}
	template <typename Dtype>
	CrossEntropyLossOp<Dtype>::CrossEntropyLossOp(CrossEntropyLossOpParam param)
	{
		op_type_ = "CrossEntropyLoss";
		param_ = param;
		labels_ = NULL;
	}
	template <typename Dtype>
	CrossEntropyLossOp<Dtype>::~CrossEntropyLossOp()
	{
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		return 0;
	}
	template <typename Dtype>
	std::string CrossEntropyLossOp<Dtype>::genOpParamStr() const
	{
		return "";
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{

		return 0;
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		return 0;
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		
		return 0;
	}
	template <typename Dtype>
	void CrossEntropyLossOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		if (next[1]->getShape()[tind::eNum] != prev[0]->getShape()[tind::eNum] || next[1]->getShape()[tind::eChannels] != 1)
		{
			DLOG_ERR("[ CrossEntropyLossOp::forward ]: label's size is not match\n");
			return;
		}

		const int outputSize4D = prev[0]->getSize()[tind::e4D];
		const int outputSize3D = prev[0]->getSize()[tind::e3D];
		const std::vector<int> outputShape = prev[0]->getShape();

		// recheck member labels_
		if (labels_ == NULL || labels_->getSize()[tind::e4D] != outputSize4D)
			labels_.reset(new Tensor<Dtype>(prev[0]->getShape()));

		// convert orgLabel format for classification task, save result in labels_
		const Dtype* orgLabelData = (Dtype *)next[1]->getData();
		//for (int j = 0; j < next[1]->getSize()[tind::e4D]; j++)
		//	printf("%f, ", orgLabelData[j]);

		Dtype* labelData = (Dtype *)labels_->getData();
		memset(labelData, 0, sizeof(Dtype)*outputSize4D);

		//printf("%f, %d\n", orgLabelData[0], next.size());
		const int classNum = labels_->getShape()[1];	//channels = class num
		for (int i = 0; i < labels_->getShape()[0]; i++)
			labelData[i * classNum + (int)orgLabelData[i]] = 1;

		// Pay attention: In caffe, CrossEntropyLoss in SigmoidCrossEntropyLossLayer 
		//                is not similar with the origin formula. Please refer to
		//				  http://blog.csdn.net/u012235274/article/details/51361290
		// compute loss (for softmax)
		const Dtype* outputData = (Dtype *)prev[0]->getData();
		Dtype loss = 0.0f;

		//for (int batchId = 0; batchId < outputShape[tind::eNum]; batchId++)	// original type
		//	for (int i = 0; i < outputSize3D; i++)
		//			loss -= labelData[batchId*outputSize3D + i] * std::log(std::max(outputData[batchId*outputSize3D + i], Dtype(FLT_MIN)));

		for (int i = 0; i < outputSize4D; i++)
			if (labelData[i] != 0)
				loss -= labelData[i] * std::log(std::max(outputData[i], Dtype(FLT_MIN)));

		*(Dtype *)next[2]->getData() = loss / outputShape[tind::eNum];
	}

	template <typename Dtype>
	void CrossEntropyLossOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		// get diff: input(lastOutput/label), output(lastDiff)
		//printf("start CrossEntropyLossOp backward:(%d, %d, %d, %d)\n", prev.size(), next.size(), prevDiff.size(), nextDiff.size());
		prevDiff[0]->setZero();

		// labels_ should be setted in forward operation, and in backward, it needn't to be converted again
		const int outputSize4D = next[0]->getSize()[tind::e4D];
		if (labels_ == NULL || labels_->getSize()[tind::e4D] != outputSize4D)
		{
			DLOG_ERR("[ CrossEntropyLossOp::backward ]: labels_ is invalid \n");
			return ;
		}

		Dtype* labelData = (Dtype *)labels_->getData();
		
		const int labelsSize3D = labels_->getSize()[tind::e3D];
		const int outputSize3D = next[0]->getSize()[tind::e3D];
		const int diffSize3D = prevDiff[0]->getSize()[tind::e3D];

		Dtype* labelDataBase = (Dtype *)labels_->getData();
		Dtype* outputDataBase = (Dtype *)next[0]->getData();
		for (int on = 0; on < next[0]->getShape()[0]; on++)
		{
			const Dtype* labelData = labelDataBase + on * labelsSize3D;
			const Dtype* outputData = outputDataBase + on * outputSize3D;
			Dtype* diffData = (Dtype *)prevDiff[0]->getData() + on * diffSize3D;
			for (int nextDiffIdx = 0; nextDiffIdx < diffSize3D; nextDiffIdx++)
			{
				const int dataIdx = nextDiffIdx;
				diffData[nextDiffIdx] -= ((labelData[dataIdx] / (outputData[dataIdx])));
			}
		}
	}

	INSTANTIATE_CLASS(CrossEntropyLossOp);

}//namespace
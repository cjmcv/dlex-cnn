////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/softmax_op.h"
#include <algorithm>
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	SoftmaxOp<Dtype>::SoftmaxOp()
	{
		op_type_ = "Softmax";
	}
	template <typename Dtype>
	SoftmaxOp<Dtype>::SoftmaxOp(SoftmaxOpParam param)
	{
		op_type_ = "Softmax";
		param_ = param;
	}
	template <typename Dtype>
	SoftmaxOp<Dtype>::~SoftmaxOp()
	{

	}
	template <typename Dtype>
	int SoftmaxOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		return 0;
	}
	template <typename Dtype>
	std::string SoftmaxOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int SoftmaxOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape = inShape;
		return 0;
	}
	template <typename Dtype>
	int SoftmaxOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxOp::allocBuf4Node ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		return 0;
	}

	template <typename Dtype>
	int SoftmaxOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		//data.clear();
		//data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		return 0;
	}

	template <typename Dtype>
	void SoftmaxOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prevDataSize = prev[0]->getSize();
		const std::vector<int> nextDataSize = next[0]->getSize();
		//const std::vector<int> prevDataShape = prev[0]->getShape();
		const std::vector<int> nextDataShape = next[0]->getShape();

		Dtype *prevDataBase = (Dtype *)prev[0]->getData();
		Dtype *nextDataBase = (Dtype *)next[0]->getData();

		const int nextDataNum = nextDataShape[tind::eNum];
		const int prevDataSize3D = prevDataSize[tind::e3D];
		const int nextDataSize3D = nextDataSize[tind::e3D];
		for (int nn = 0; nn < nextDataNum; nn++)
		{
			const Dtype* prevData = prevDataBase + nn * prevDataSize3D;// *sizeof(float);
			Dtype* nextData = nextDataBase + nn * nextDataSize3D;// *sizeof(float);

			//step1 : find max value
			Dtype maxVal = prevData[0];
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				maxVal = std::max(maxVal, prevData[prevDataIdx]);
			}
			//step2 : sum
			Dtype sum = 0;
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				nextData[prevDataIdx] = std::exp(prevData[prevDataIdx] - maxVal);
				sum += nextData[prevDataIdx];
			}
			//step3 : div
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				nextData[prevDataIdx] = nextData[prevDataIdx] / sum;
			}
		}
	}

	template <typename Dtype>
	void SoftmaxOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		const std::vector<int> prevDataSize = prev[0]->getSize();
		const std::vector<int> nextDataSize = next[0]->getSize();
		const std::vector<int> prevDiffSize = prevDiff[0]->getSize();
		const std::vector<int> nextDiffSize = nextDiff[0]->getSize();

		const std::vector<int> prevDataShape = prev[0]->getShape();
		const std::vector<int> nextDataShape = next[0]->getShape();
		const std::vector<int> prevDiffShape = prevDiff[0]->getShape();
		const std::vector<int> nextDiffShape = nextDiff[0]->getShape();

		Dtype *prevDataBase = (Dtype *)prev[0]->getData();
		Dtype *nextDataBase = (Dtype *)next[0]->getData();
		Dtype *prevDiffBase = (Dtype *)prevDiff[0]->getData();
		Dtype *nextDiffBase = (Dtype *)nextDiff[0]->getData();

		if (prevDataSize[tind::e4D] != nextDataSize[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input and output data must be equal \n");
			return;
		}
		if (prevDiffSize[tind::e4D] != nextDiffSize[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output diff must be equal \n");
			return;
		}
		if (prevDiffSize[tind::e4D] != prevDataSize[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output data must be equal \n");
			return;
		}

		//update prevDiff
		prevDiff[0]->setZero();
		const int prevDataSize3D = prevDataSize[tind::e3D];
		const int nextDataSize3D = nextDataSize[tind::e3D];
		const int prevDiffSize3D = prevDiffSize[tind::e3D];
		const int nextDiffSize3D = nextDiffSize[tind::e3D];
		for (int pn = 0; pn < prevDataShape[tind::eNum]; pn++)
		{
			const Dtype* prevData = prevDataBase + pn * prevDataSize3D;
			const Dtype* nextData = nextDataBase + pn * nextDataSize3D;
			const Dtype* nextDiffData = nextDiffBase + pn * nextDiffSize3D;
			Dtype* prevDiffData = prevDiffBase + pn * prevDiffSize3D;
			for (int prevDiffIdx = 0; prevDiffIdx < prevDiffSize3D; prevDiffIdx++)
			{
				for (int nextDiffIdx = 0; nextDiffIdx < nextDiffSize3D; nextDiffIdx++)
				{
					if (nextDiffIdx == prevDiffIdx)
					{
						prevDiffData[prevDiffIdx] += nextData[prevDiffIdx] * (1.0f - nextData[prevDiffIdx]) * nextDiffData[nextDiffIdx];
					}
					else
					{
						prevDiffData[prevDiffIdx] -= nextData[prevDiffIdx] * nextData[nextDiffIdx] * nextDiffData[nextDiffIdx];
					}
				}
			}
		}
		//update this layer's param
		//softmax layer : nop
	}

	INSTANTIATE_CLASS(SoftmaxOp);

}//namespace
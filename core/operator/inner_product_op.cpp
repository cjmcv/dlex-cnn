////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/inner_product_op.h"
#include "util/math_functions.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	InnerProductOp<Dtype>::InnerProductOp()
	{
		op_type_ = "InnerProduct";
	}
	template <typename Dtype>
	InnerProductOp<Dtype>::InnerProductOp(InnerProductOpParam param)
	{
		op_type_ = "InnerProduct";
		param_ = param;
	}
	template <typename Dtype>
	InnerProductOp<Dtype>::~InnerProductOp()
	{

	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.blas_enable = atoi(fetchSubStr(optStr, "blas_enable:", ",").c_str());
		param_.num_hidden = atoi(fetchSubStr(optStr, "num_hidden:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string InnerProductOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "blas_enable:" << param_.blas_enable << ",num_hidden:" << param_.num_hidden << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape.clear();
		outShape.push_back(inShape[tind::eNum]);
		outShape.push_back(param_.num_hidden);
		outShape.push_back(1);
		outShape.push_back(1);
		return 0;
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape, 
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		data.clear();
		printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());

		int inShape3DSize = inShape[1] * inShape[2] * inShape[3];
		int outShape3DSize = outShape[1] * outShape[2] * outShape[3];

		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		//weight
		data.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		normal_distribution_init<Dtype>((Dtype *)data[1]->getData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		//blas
		if (param_.blas_enable)
		{
			data.push_back(std::make_shared<Tensor<Dtype>>(1, outShape[1], 1, 1));
			dlex_set<Dtype>((Dtype *)data[2]->getData(), data[2]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}
	template <typename Dtype>
	int InnerProductOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n", 
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}
		if (outShape[tind::eNum] <= 0 || outShape[tind::eChannels] <= 0 || 
			outShape[tind::eHeight] != 1 || outShape[tind::eWidth] != 1 ||
			outShape[tind::eNum] > 50000 || outShape[tind::eChannels] > 50000)
		{
			DLOG_ERR("[ InnerProductOp::allocOpBuf4Train ]: outShape is invalid -> (%d, %d, %d, %d) \n",
				outShape[tind::eNum], outShape[tind::eChannels], outShape[tind::eHeight], outShape[tind::eWidth]);
			return -1;
		}
		
		//data.clear();
		gradient_.clear();
		diff_.clear();
		
		//printf("data and gradient: size() : %d, %d\n", data.size(), gradient_.size());

		int inShape3DSize = inShape[1] * inShape[2] * inShape[3];
		int outShape3DSize = outShape[1] * outShape[2] * outShape[3];
		
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		//data.push_back(std::make_shared<Tensor<Dtype>>(inShape));
		//data.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		//normal_distribution_init<Dtype>((Dtype *)data[1]->getData(), data[1]->getSize()[tind::e4D], 0.0f, 0.1f);

		gradient_.push_back(std::make_shared<Tensor<Dtype>>(1, inShape3DSize * outShape3DSize, 1, 1));
		dlex_set<Dtype>((Dtype *)gradient_[0]->getData(), gradient_[0]->getSize()[tind::e4D], 0.0f);

		if (param_.blas_enable)
		{
			//data.push_back(std::make_shared<Tensor<Dtype>>(1, outShape[1], 1, 1));
			//dlex_set<Dtype>((Dtype *)data[2]->getData(), data[2]->getSize()[tind::e4D], 0.0f);

			gradient_.push_back(std::make_shared<Tensor<Dtype>>(1, outShape[1], 1, 1));
			dlex_set<Dtype>((Dtype *)gradient_[1]->getData(), gradient_[1]->getSize()[tind::e4D], 0.0f);
		}
		return 0;
	}

	template <typename Dtype>
	void InnerProductOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		//printf("InnerProductOp: prev = %d, next = %d\n", prev.size(), next.size());
		const std::vector<int> prevDataSize = prev[0]->getSize();
		const std::vector<int> nextDataSize = next[0]->getSize();

		//printf("into innerProduct forward:(%d, %d), (%d, %d)\n", prev3DSize, next3DSize, prev[1]->get4DSize(), prev[2]->get4DSize());

		const Dtype* prevData = (Dtype *)prev[0]->getData();
		Dtype* nextData = (Dtype *)next[0]->getData();
		const Dtype* weightData = (Dtype *)prev[1]->getData();
		const Dtype* biasData = param_.blas_enable ? (Dtype *)prev[2]->getData() : nullptr;

		//for (int i = 0; i < prev[2]->get4DSize(); i++)
		//{
		//	printf("%f, ", biasData[i]);
		//}

		auto worker = [&](const int start, const int stop){
			gemm(false, true, stop - start, nextDataSize[tind::e3D], prevDataSize[tind::e3D], 1.0, prevData + start * prevDataSize[tind::e3D], weightData, 0.0, nextData + start * nextDataSize[tind::e3D]);
		};
		//dispatch_worker(worker,prevSize.number);
		worker(0, prev[0]->getShape()[tind::eNum]);

		if (param_.blas_enable)
			add_bias(prev[0]->getShape()[tind::eNum], nextDataSize[tind::e3D], biasData, nextData);

		//// The method for adding bias to data in caffe
		//Dtype *bias_multiplier = (Dtype *)malloc(sizeof(Dtype) * prev[0]->getShape()[tind::eNum]);
		//for (int i = 0; i < prev[0]->getShape()[tind::eNum]; i++)
		//	bias_multiplier[i] = 1;
		//auto worker2 = [&](const int start, const int stop){
		//	gemm(false, false, stop - start, nextDataSize[tind::e3D], 1, 1.0, bias_multiplier, biasData, 1.0, nextData + start * nextDataSize[tind::e3D]);
		//};
		//worker2(0, prev[0]->getShape()[0]);

	}

	template <typename Dtype>
	void InnerProductOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		const Dtype* prevData = (Dtype*)prev[0]->getData();
		const Dtype* nextData = (Dtype*)next[0]->getData();
		Dtype* prevDiffData = (Dtype*)prevDiff[0]->getData();
		const Dtype* nextDiffData = (Dtype*)nextDiff[0]->getData();
		const Dtype* weightData = (Dtype*)prev[1]->getData();
		//const Dtype* biasData = param_.blas_enable ? (Dtype*)prev[2]->getData() : nullptr;

		const std::vector<int> prevDiffSize = prevDiff[0]->getSize();
		const std::vector<int> prevDataSize = prev[0]->getSize();
		const std::vector<int> nextDiffSize = nextDiff[0]->getSize();
		const std::vector<int> nextDataSize = next[0]->getSize();

		const std::vector<int> weightSize = prev[1]->getSize();

		const std::vector<int> prevDataShape = prev[0]->getShape();

		const std::vector<int> nextDataShape = next[0]->getShape();
		//const std::vector<int> nextDiffShape = nextDiff[0]->getShape();

		if (nextDataShape[tind::eHeight] != 1 || nextDataShape[tind::eWidth] != 1)
		{
			DLOG_ERR("[ InnerProductOp::backward ]: using channels as label only, height and width shoule be 1 \n");
			return;
		}
		if (weightSize[tind::e4D] != prevDataSize[tind::e3D] * nextDataSize[2])
		{
			DLOG_ERR("[ InnerProductOp::backward ]: weightSize is invalidate!\n");
			return;
		}
		if (param_.blas_enable)
		{
			if (prev[2]->getSize()[tind::e4D] != nextDataSize[tind::e3D])
			{
				DLOG_ERR("[ InnerProductOp::backward ]: bias size is invalidate!\n");
				return;
			}
		}
		if (prevDiffSize[tind::e4D] != prevDataSize[tind::e4D])
		{
			DLOG_ERR("[ InnerProductOp::backward ]: the size of prevDiff and prev must be equal\n");
			return;
		}

		////////////////////////////////////////////////////////////////////////////////////////
		//update prevDiff
		// prevDiff(num, in3DSize) = nextDiff(num, hidden_num) * weight(hidden_num, in3DSize)
		// -> prevDiff(num, prevDiffSize[tind::e3D]) = nextDiff(num, nextDiffSize[tind::e3D]) * weight(nextDiffSize[tind::e3D], in3DSize)
		auto worker = [&](const int start, const int stop){
			gemm(false, false, stop - start, prevDiffSize[tind::e3D], nextDiffSize[tind::e3D], 
				1.0, nextDiffData + start * nextDiffSize[tind::e3D], weightData, 
				0.0, prevDiffData + start * prevDiffSize[tind::e3D]);
		};
		//dispatch_worker(worker, prevSize.number);
		worker(0, prevDataShape[tind::eNum]);

		
		////////////////////////////////////////////////////////////////////////////
		//update this layer's param
		//get weight gradient
		Dtype* weightGradientData = (Dtype *)gradient_[0]->getData();

		// nextDiff(num, hidden_num) -> nextDiff'(hidden_num, num)
		// O(M,N) = weightGradient(hidden_num, in3DSize) = nextDiff'(hidden_num, num) * prevData(num, in3DSize)
		// -> M=hidden_num, N=in3DSize, K=num
		auto worker2 = [&](const int start, const int stop){
			gemm(true, false, nextDiffSize[tind::e3D], prevDataSize[tind::e3D], prevDataShape[tind::eNum],
				1.0, nextDiffData, prevData,
				1.0, weightGradientData);	//1.0
		};
		//dispatch_worker(worker, prevSize.number);
		worker2(0, prevDataShape[tind::eNum]);

		//div by batch size
		div_inplace(weightGradientData, (Dtype)nextDataShape[tind::eNum], weightSize[tind::e4D]);

		////////////////////////////////////////////////////////////////////////
		//update bias
		if (param_.blas_enable)
		{
			//get bias diff	
			Dtype* biasGradientData = (Dtype *)gradient_[1]->getData();
			const std::vector<int> biasGradSize = gradient_[1]->getSize();

			gradient_[1]->setZero();
			backward_bias(nextDataShape[tind::eNum], biasGradSize[tind::e3D], nextDiffData, biasGradientData);

			//div by batch size
			div_inplace(biasGradientData, (Dtype)nextDataShape[tind::eNum], biasGradSize[tind::e4D]);
		}

	}

	INSTANTIATE_CLASS(InnerProductOp);

}//namespace
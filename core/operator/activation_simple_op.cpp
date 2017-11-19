////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/activation_simple_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	ActivationOp<Dtype>::ActivationOp()
	{
		op_type_ = "Activation";
	}
	template <typename Dtype>
	ActivationOp<Dtype>::ActivationOp(ActivationOpParam param)
	{
		op_type_ = "Activation";
		param_ = param;
		setOpFunc();
	}
	template <typename Dtype>
	ActivationOp<Dtype>::~ActivationOp()
	{

	}
	template <typename Dtype>
	int ActivationOp<Dtype>::setOpFunc()
	{
		switch (param_.activationType) {
		case tind::Activation::eReLU:
			pAct = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
			pRevAct = std::bind(&ActivationOp::rev_relu, this, std::placeholders::_1, std::placeholders::_2);
			break;
		case tind::Activation::eSigmoid:
			pAct = std::bind(&ActivationOp::sigmoid, this, std::placeholders::_1);
			pRevAct = std::bind(&ActivationOp::rev_sigmoid, this, std::placeholders::_1, std::placeholders::_2);
			break;
		case tind::Activation::eTanh:
			pAct = std::bind(&ActivationOp::tanh, this, std::placeholders::_1);
			pRevAct = std::bind(&ActivationOp::rev_tanh, this, std::placeholders::_1, std::placeholders::_2);
			break;
		default:
			pAct = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
			pRevAct = std::bind(&ActivationOp::rev_relu, this, std::placeholders::_1, std::placeholders::_2);
		}

		return 0;
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.activationType = (tind::Activation)atoi(fetchSubStr(optStr, "activationType:", ",").c_str());

		setOpFunc();
		return 0;
	}
	template <typename Dtype>
	std::string ActivationOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "activationType:" << param_.activationType << ",negative_slope:" << param_.negative_slope << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape = inShape;
		return 0;
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InputOp::allocBuf4Node ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		return 0;
	}

	template <typename Dtype>
	int ActivationOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InputOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		return 0;
	}

	template <typename Dtype>
	void ActivationOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		//pAct = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
		//const std::vector<int> prevShape = prev[0]->getShape();
		const std::vector<int> prevSize = prev[0]->getSize();
		const std::vector<int> nextSize = next[0]->getSize();
		float* prevData = (float *)prev[0]->getData();
		float* nextData = (float *)next[0]->getData();

		for (int n = 0; n < prev[0]->getShape()[tind::eNum]; n++)
		{
			float* prevData_n = prevData + n * prevSize[tind::e3D];
			float* nextData_n = nextData + n * nextSize[tind::e3D];
			for (int i = 0; i < prevSize[tind::e3D]; i++)
			{
				nextData_n[i] = pAct(prevData_n[i]);
			}
		}
	}

	template <typename Dtype>
	void ActivationOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		const int prevSize3D = prev[0]->getSize()[tind::e3D];
		const int nextSize3D = next[0]->getSize()[tind::e3D];
		const int prevDiffSize3D = prevDiff[0]->getSize()[tind::e3D];
		const int nextDiffSize3D = nextDiff[0]->getSize()[tind::e3D];
		float* prevData = (float *)prev[0]->getData();
		float* nextData = (float *)next[0]->getData();
		float* prevDiffData = (float *)prevDiff[0]->getData();
		float* nextDiffData = (float *)nextDiff[0]->getData();

		float* act_x = NULL;
		int act_len = 0;
		switch (param_.activationType) {
		case tind::Activation::eReLU:
			act_x = prevData;
			act_len = prevSize3D;
			break;
		default:
			act_x = nextData;
			act_len = nextSize3D;
		}

		for (int n = 0; n < prev[0]->getShape()[tind::eNum]; n++)
		{
			float* actx_n = act_x + n * act_len;
			float* prevDiffData_n = prevDiffData + n * prevDiffSize3D;
			float* nextDiffData_n = nextDiffData + n * nextDiffSize3D;
			for (int i = 0; i < act_len; i++)
			{
				pRevAct(actx_n[i], nextDiffData_n[i]);
			}
		}
	}

	INSTANTIATE_CLASS(ActivationOp);
}//namespace
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/input_op.h"
#include <sstream>

namespace dlex_cnn
{	
	template <typename Dtype>
	InputOp<Dtype>::InputOp()
	{
		op_type_ = "Input";
	}
	template <typename Dtype>
	InputOp<Dtype>::InputOp(InputOpParam param)
	{
		op_type_ = "Input";
		param_ = param;
	}
	template <typename Dtype>
	InputOp<Dtype>::~InputOp()
	{

	}
	template <typename Dtype>
	int InputOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.num = atoi(fetchSubStr(optStr, "num:", ",").c_str());
		param_.channels = atoi(fetchSubStr(optStr, "channels:", ",").c_str());
		param_.height = atoi(fetchSubStr(optStr, "height:", ",").c_str());
		param_.width = atoi(fetchSubStr(optStr, "width:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string InputOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "num:" << param_.num << ",channels:" << param_.channels << ",height:" << param_.height << ",width:" << param_.width << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int InputOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		// if input shape has not been setted
		if (inShape.size() != 4)
		{
			inShape.clear();
			inShape.push_back(param_.num);
			inShape.push_back(param_.channels);
			inShape.push_back(param_.height);
			inShape.push_back(param_.width);
		}
		else if (inShape[1] != param_.channels || 
			inShape[2] != param_.height || 
			inShape[3] != param_.width)
		{
			DLOG_ERR("[ InputOp::inferOutShape ]: inShape[1] != param_.channels_ || \
				inShape[2] != param_.height_ || inShape[3] != param_.width_ \n");

			return -1;
		}
		else
			param_.num = inShape[0];

		outShape = inShape;
		return 0;
	}
	template <typename Dtype>
	int InputOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
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
	int InputOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
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

		//data.clear();
		//data.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		return 0;
	}

	template <typename Dtype>
	void InputOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		for (int i = 0; i < prev.size(); i++)
			prev[i]->copyDataTo(*next[i]);
	}

	template <typename Dtype>
	void InputOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		//the backward operation in InputOp is empty
	}

	INSTANTIATE_CLASS(InputOp);
}//namespace
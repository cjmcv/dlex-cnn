////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/output_op.h"
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	OutputOp<Dtype>::OutputOp()
	{
		param_.label_dim = 1;
		op_type_ = "Output";
	}
	template <typename Dtype>
	OutputOp<Dtype>::OutputOp(OutputOpParam param)
	{
		op_type_ = "Output";
		param_ = param;
	}
	template <typename Dtype>
	OutputOp<Dtype>::~OutputOp()
	{

	}
	template <typename Dtype>
	int OutputOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		std::string optStr = opParamStr;
		param_.label_dim = atoi(fetchSubStr(optStr, "label_dim:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string OutputOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << "label_dim:" << param_.label_dim << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int OutputOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape = inShape;
		return 0;
	}
	template <typename Dtype>
	int OutputOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ OutputOp::allocBuf4Node ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);

			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));	// output data
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape[0], param_.label_dim, 1, 1));	// label
		data.push_back(std::make_shared<Tensor<Dtype>>(1, 1, 1, 1)); //loss

		return 0;
	}

	template <typename Dtype>
	int OutputOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ OutputOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n",
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
	void OutputOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		// the forward operation in output node should not be called
	}

	template <typename Dtype>
	void OutputOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		// the backward operation in output node should not be called
	}

	INSTANTIATE_CLASS(OutputOp);
}//namespace
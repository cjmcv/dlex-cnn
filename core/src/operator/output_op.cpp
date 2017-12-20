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
	int OutputOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		std::string opt_str = op_param_str;
		param_.label_dim = atoi(fetchSubStr(opt_str, "label_dim:", ",").c_str());

		return 0;
	}
	template <typename Dtype>
	std::string OutputOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << "label_dim:" << param_.label_dim << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int OutputOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		out_shape = in_shape;
		return 0;
	}
	template <typename Dtype>
	int OutputOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ OutputOp::allocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);

			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));	// output data
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape[0], param_.label_dim, 1, 1));	// label
		data.push_back(std::make_shared<Tensor<Dtype>>(1, 1, 1, 1)); //loss

		return 0;
	}

	template <typename Dtype>
	int OutputOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ OutputOp::allocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		//data.clear();
		//data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		return 0;
	}

	INSTANTIATE_CLASS(OutputOp);
}//namespace
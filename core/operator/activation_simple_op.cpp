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
		switch (param_.activation_type) {
		case tind::Activation::eReLU:
			p_act = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
			p_rev_act = std::bind(&ActivationOp::rev_relu, this, std::placeholders::_1, std::placeholders::_2);
			break;
		case tind::Activation::eSigmoid:
			p_act = std::bind(&ActivationOp::sigmoid, this, std::placeholders::_1);
			p_rev_act = std::bind(&ActivationOp::rev_sigmoid, this, std::placeholders::_1, std::placeholders::_2);
			break;
		case tind::Activation::eTanh:
			p_act = std::bind(&ActivationOp::tanh, this, std::placeholders::_1);
			p_rev_act = std::bind(&ActivationOp::rev_tanh, this, std::placeholders::_1, std::placeholders::_2);
			break;
		default:
			p_act = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
			p_rev_act = std::bind(&ActivationOp::rev_relu, this, std::placeholders::_1, std::placeholders::_2);
		}

		return 0;
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		std::string opt_str = op_param_str;
		param_.activation_type = (tind::Activation)atoi(fetchSubStr(opt_str, "activation_type:", ",").c_str());
		param_.negative_slope = (tind::Activation)atoi(fetchSubStr(opt_str, "negative_slope:", ",").c_str());

		setOpFunc();
		return 0;
	}
	template <typename Dtype>
	std::string ActivationOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << "activation_type:" << param_.activation_type << ",negative_slope:" << param_.negative_slope << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		out_shape = in_shape;
		return 0;
	}
	template <typename Dtype>
	int ActivationOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InputOp::allocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		return 0;
	}

	template <typename Dtype>
	int ActivationOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ InputOp::allocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		return 0;
	}

	template <typename Dtype>
	void ActivationOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		//p_act = std::bind(&ActivationOp::relu, this, std::placeholders::_1);
		//const std::vector<int> prev_shape = prev[0]->getShape();
		const std::vector<int> prev_size = prev[0]->getSize();
		const std::vector<int> next_size = next[0]->getSize();
		float* prev_data = (float *)prev[0]->getCpuData();
		float* next_data = (float *)next[0]->getCpuData();

		for (int n = 0; n < prev[0]->getShape()[tind::eNum]; n++)
		{
			float* prev_data_n = prev_data + n * prev_size[tind::e3D];
			float* next_data_n = next_data + n * next_size[tind::e3D];
			for (int i = 0; i < prev_size[tind::e3D]; i++)
			{
				next_data_n[i] = p_act(prev_data_n[i]);
			}
		}
	}

	template <typename Dtype>
	void ActivationOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		const int prev_size3D = prev[0]->getSize()[tind::e3D];
		const int next_size3D = next[0]->getSize()[tind::e3D];
		const int prev_diff_size3D = prev_diff[0]->getSize()[tind::e3D];
		const int next_diff_size3D = next_diff[0]->getSize()[tind::e3D];
		float* prev_data = (float *)prev[0]->getCpuData();
		float* next_data = (float *)next[0]->getCpuData();
		float* prev_diff_data = (float *)prev_diff[0]->getCpuData();
		float* next_diff_data = (float *)next_diff[0]->getCpuData();

		float* act_x = NULL;
		int act_len = 0;
		switch (param_.activation_type) {
		case tind::Activation::eReLU:
			act_x = prev_data;
			act_len = prev_size3D;
			break;
		default:
			act_x = next_data;
			act_len = next_size3D;
		}

		for (int n = 0; n < prev[0]->getShape()[tind::eNum]; n++)
		{
			float* actx_n = act_x + n * act_len;
			float* prev_diff_data_n = prev_diff_data + n * prev_diff_size3D;
			float* next_diff_data_n = next_diff_data + n * next_diff_size3D;
			for (int i = 0; i < act_len; i++)
			{
				p_rev_act(actx_n[i], next_diff_data_n[i]);
			}
		}
	}

	INSTANTIATE_CLASS(ActivationOp);
}//namespace
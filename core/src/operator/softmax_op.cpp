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
	int SoftmaxOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		return 0;
	}
	template <typename Dtype>
	std::string SoftmaxOp<Dtype>::genOpParamStr() const
	{
		std::stringstream param_str;
		param_str << ",";
		return param_str.str();
	}
	template <typename Dtype>
	int SoftmaxOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{
		out_shape = in_shape;
		return 0;
	}
	template <typename Dtype>
	int SoftmaxOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxOp::allocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		return 0;
	}

	template <typename Dtype>
	int SoftmaxOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
	{
		if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
			in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
			in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
			in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxOp::allocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
				in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
			return -1;
		}

		//data.clear();
		//data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

		return 0;
	}

	template <typename Dtype>
	void SoftmaxOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();
		//const std::vector<int> prev_data_shape = prev[0]->getShape();
		const std::vector<int> next_data_shape = next[0]->getShape();

		Dtype *prevDataBase = (Dtype *)prev[0]->getCpuData();
		Dtype *nextDataBase = (Dtype *)next[0]->getCpuData();

		const int nextDataNum = next_data_shape[tind::eNum];
		const int prevDataSize3D = prev_data_size[tind::e3D];
		const int nextDataSize3D = next_data_size[tind::e3D];
		for (int nn = 0; nn < nextDataNum; nn++)
		{
			const Dtype* prev_data = prevDataBase + nn * prevDataSize3D;// *sizeof(float);
			Dtype* next_data = nextDataBase + nn * nextDataSize3D;// *sizeof(float);

			//step1 : find max value
			Dtype maxVal = prev_data[0];
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				maxVal = std::max(maxVal, prev_data[prevDataIdx]);
			}
			//step2 : sum
			Dtype sum = 0;
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				next_data[prevDataIdx] = std::exp(prev_data[prevDataIdx] - maxVal);
				sum += next_data[prevDataIdx];
			}
			//step3 : div
			for (int prevDataIdx = 0; prevDataIdx < prevDataSize3D; prevDataIdx++)
			{
				next_data[prevDataIdx] = next_data[prevDataIdx] / sum;
			}
		}
	}

	template <typename Dtype>
	void SoftmaxOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		const std::vector<int> prev_data_size = prev[0]->getSize();
		const std::vector<int> next_data_size = next[0]->getSize();
		const std::vector<int> prev_diff_size = prev_diff[0]->getSize();
		const std::vector<int> next_diff_size = next_diff[0]->getSize();

		const std::vector<int> prev_data_shape = prev[0]->getShape();
		const std::vector<int> next_data_shape = next[0]->getShape();
		const std::vector<int> prev_diff_shape = prev_diff[0]->getShape();
		const std::vector<int> next_diff_shape = next_diff[0]->getShape();

		Dtype *prevDataBase = (Dtype *)prev[0]->getCpuData();
		Dtype *nextDataBase = (Dtype *)next[0]->getCpuData();
		Dtype *prevDiffBase = (Dtype *)prev_diff[0]->getCpuData();
		Dtype *nextDiffBase = (Dtype *)next_diff[0]->getCpuData();

		if (prev_data_size[tind::e4D] != next_data_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input and output data must be equal \n");
			return;
		}
		if (prev_diff_size[tind::e4D] != next_diff_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output diff must be equal \n");
			return;
		}
		if (prev_diff_size[tind::e4D] != prev_data_size[tind::e4D])
		{
			DLOG_ERR("[ SoftmaxOp::backward ]: the size of input diff and output data must be equal \n");
			return;
		}

		//update prev_diff
		prev_diff[0]->setCpuZero();
		const int prevDataSize3D = prev_data_size[tind::e3D];
		const int nextDataSize3D = next_data_size[tind::e3D];
		const int prev_diff_size3D = prev_diff_size[tind::e3D];
		const int next_diff_size3D = next_diff_size[tind::e3D];
		for (int pn = 0; pn < prev_data_shape[tind::eNum]; pn++)
		{
			const Dtype* prev_data = prevDataBase + pn * prevDataSize3D;
			const Dtype* next_data = nextDataBase + pn * nextDataSize3D;
			const Dtype* next_diff_data = nextDiffBase + pn * next_diff_size3D;
			Dtype* prev_diff_data = prevDiffBase + pn * prev_diff_size3D;
			for (int prevDiffIdx = 0; prevDiffIdx < prev_diff_size3D; prevDiffIdx++)
			{
				for (int next_diff_idx = 0; next_diff_idx < next_diff_size3D; next_diff_idx++)
				{
					if (next_diff_idx == prevDiffIdx)
					{
						prev_diff_data[prevDiffIdx] += next_data[prevDiffIdx] * (1.0f - next_data[prevDiffIdx]) * next_diff_data[next_diff_idx];
					}
					else
					{
						prev_diff_data[prevDiffIdx] -= next_data[prevDiffIdx] * next_data[next_diff_idx] * next_diff_data[next_diff_idx];
					}
				}
			}
		}
		//update this layer's param
		//softmax layer : nop
	}

	INSTANTIATE_CLASS(SoftmaxOp);

}//namespace
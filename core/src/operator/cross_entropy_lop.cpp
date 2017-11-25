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
	int CrossEntropyLossOp<Dtype>::setOpParam(const std::string &op_param_str)
	{
		return 0;
	}
	template <typename Dtype>
	std::string CrossEntropyLossOp<Dtype>::genOpParamStr() const
	{
		return "";
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::inferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape)
	{

		return 0;
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::allocBuf4Node(const std::vector<int> &in_shape,
		const std::vector<int> &out_shape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		return 0;
	}
	template <typename Dtype>
	int CrossEntropyLossOp<Dtype>::allocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape)
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

		const int output_size4D = prev[0]->getSize()[tind::e4D];
		const int output_size3D = prev[0]->getSize()[tind::e3D];
		const std::vector<int> output_shape = prev[0]->getShape();

		// recheck member labels_
		if (labels_ == NULL || labels_->getSize()[tind::e4D] != output_size4D)
			labels_.reset(new Tensor<Dtype>(prev[0]->getShape()));

		// convert orgLabel format for classification task, save result in labels_
		const Dtype* org_label_data = (Dtype *)next[1]->getCpuData();
		//for (int j = 0; j < next[1]->getSize()[tind::e4D]; j++)
		//	printf("%f, ", org_label_data[j]);

		Dtype* label_data = (Dtype *)labels_->getCpuData();
		memset(label_data, 0, sizeof(Dtype)*output_size4D);

		//printf("%f, %d\n", org_label_data[0], next.size());
		const int class_num = labels_->getShape()[1];	//channels = class num
		for (int i = 0; i < labels_->getShape()[0]; i++)
			label_data[i * class_num + (int)org_label_data[i]] = 1;

		// Pay attention: In caffe, CrossEntropyLoss in SigmoidCrossEntropyLossLayer 
		//                is not similar with the origin formula. Please refer to
		//				  http://blog.csdn.net/u012235274/article/details/51361290
		// compute loss (for softmax)
		const Dtype* output_data = (Dtype *)prev[0]->getCpuData();
		Dtype loss = 0.0f;

		//for (int batchId = 0; batchId < output_shape[tind::eNum]; batchId++)	// original type
		//	for (int i = 0; i < output_size3D; i++)
		//			loss -= label_data[batchId*output_size3D + i] * std::log(std::max(output_data[batchId*output_size3D + i], Dtype(FLT_MIN)));

		for (int i = 0; i < output_size4D; i++)
			if (label_data[i] != 0)
				loss -= label_data[i] * std::log(std::max(output_data[i], Dtype(FLT_MIN)));

		*(Dtype *)next[2]->getCpuData() = loss / output_shape[tind::eNum];
	}

	template <typename Dtype>
	void CrossEntropyLossOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, 
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff)
	{
		// get diff: input(lastOutput/label), output(lastDiff)
		//printf("start CrossEntropyLossOp backward:(%d, %d, %d, %d)\n", prev.size(), next.size(), prev_diff.size(), next_diff.size());
		prev_diff[0]->setZero();

		// labels_ should be setted in forward operation, and in backward, it needn't to be converted again
		const int output_size4D = next[0]->getSize()[tind::e4D];
		if (labels_ == NULL || labels_->getSize()[tind::e4D] != output_size4D)
		{
			DLOG_ERR("[ CrossEntropyLossOp::backward ]: labels_ is invalid \n");
			return ;
		}

		Dtype* label_data = (Dtype *)labels_->getCpuData();
		
		const int labels_size3D = labels_->getSize()[tind::e3D];
		const int output_size3D = next[0]->getSize()[tind::e3D];
		const int diff_size3D = prev_diff[0]->getSize()[tind::e3D];

		Dtype* label_data_base = (Dtype *)labels_->getCpuData();
		Dtype* output_data_base = (Dtype *)next[0]->getCpuData();
		for (int on = 0; on < next[0]->getShape()[0]; on++)
		{
			const Dtype* label_data = label_data_base + on * labels_size3D;
			const Dtype* output_data = output_data_base + on * output_size3D;
			Dtype* diff_data = (Dtype *)prev_diff[0]->getCpuData() + on * diff_size3D;
			for (int next_diff_idx = 0; next_diff_idx < diff_size3D; next_diff_idx++)
			{
				const int data_idx = next_diff_idx;
				diff_data[next_diff_idx] -= ((label_data[data_idx] / (output_data[data_idx])));
			}
		}
	}

	INSTANTIATE_CLASS(CrossEntropyLossOp);

}//namespace
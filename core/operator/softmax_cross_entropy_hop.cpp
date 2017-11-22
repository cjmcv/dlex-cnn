////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/softmax_cross_entropy_hop.h"
#include <algorithm>
#include <sstream>

namespace dlex_cnn
{
	template <typename Dtype>
	SoftmaxCrossEntropyLossHOp<Dtype>::SoftmaxCrossEntropyLossHOp()
	{
		op_type_ = "SoftmaxCrossEntropyLossH";
	}
	template <typename Dtype>
	SoftmaxCrossEntropyLossHOp<Dtype>::SoftmaxCrossEntropyLossHOp(SoftmaxCrossEntropyLossHOpParam param)
	{
		op_type_ = "SoftmaxCrossEntropyLossH";
		param_ = param;
	}
	template <typename Dtype>
	SoftmaxCrossEntropyLossHOp<Dtype>::~SoftmaxCrossEntropyLossHOp()
	{

	}
	template <typename Dtype>
	int SoftmaxCrossEntropyLossHOp<Dtype>::setOpParam(const std::string &opParamStr)
	{
		// hop获取到hparam后，分别生成各子op对应的参数，对这些子op赋参
		return 0;
	}
	template <typename Dtype>
	std::string SoftmaxCrossEntropyLossHOp<Dtype>::genOpParamStr() const
	{
		std::stringstream paramStr;
		paramStr << ",";
		return paramStr.str();
	}
	template <typename Dtype>
	int SoftmaxCrossEntropyLossHOp<Dtype>::inferOutShape(std::vector<int> &inShape, std::vector<int> &outShape)
	{
		outShape = inShape;
		return 0;
	}
	template <typename Dtype>
	int SoftmaxCrossEntropyLossHOp<Dtype>::allocBuf4Node(const std::vector<int> &inShape,
		const std::vector<int> &outShape,
		std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::allocBuf4Node ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		data.clear();
		data.push_back(std::make_shared<Tensor<Dtype>>(inShape));
		
		return 0;
	}

	template <typename Dtype>
	int SoftmaxCrossEntropyLossHOp<Dtype>::allocOpBuf4Train(const std::vector<int> &inShape, const std::vector<int> &outShape)
	{
		if (inShape[tind::eNum] <= 0 || inShape[tind::eChannels] <= 0 ||
			inShape[tind::eHeight] <= 0 || inShape[tind::eWidth] <= 0 ||
			inShape[tind::eNum] > 5000 || inShape[tind::eChannels] > 5000 ||
			inShape[tind::eHeight] > 5000 || inShape[tind::eWidth] > 5000)
		{
			DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::allocOpBuf4Train ]: inShape is invalid -> (%d, %d, %d, %d) \n",
				inShape[tind::eNum], inShape[tind::eChannels], inShape[tind::eHeight], inShape[tind::eWidth]);
			return -1;
		}

		diff_.clear();
		diff_.push_back(std::make_shared<Tensor<Dtype>>(inShape));

		if (sub_ops_.size() == 0)
		{
			// can set sub_ops's params by loading inteOp's params
			std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Softmax");
			//dlex_cnn::SoftmaxOpParam softmaxParam;
			//dynamic_cast<dlex_cnn::SoftmaxOp<Dtype> *>(sm_s.get())->setOpParam(softmaxParam);

			std::shared_ptr<dlex_cnn::Op<Dtype>> cel_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("CrossEntropyLoss");
			//dlex_cnn::CrossEntropyLossOpParam CELParam;
			//dynamic_cast<dlex_cnn::CrossEntropyLossOp<Dtype> *>(cel_s.get())->setOpParam(CELParam);

			sub_ops_.push_back(sm_s);
			sub_ops_.push_back(cel_s);
		}

		return 0;
	}

	template <typename Dtype>
	void SoftmaxCrossEntropyLossHOp<Dtype>::forward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next)
	{
		//printf("start SoftmaxCrossEntropyLossHOp\n");
		int opInd[2] = { -1 };
		bool opFlag = false;
		if (sub_ops_[0]->getOpType() == "Softmax" && sub_ops_[1]->getOpType() == "CrossEntropyLoss")
		{
			opInd[0] = 0;
			opInd[1] = 1;
			opFlag = true;
		}
		else if (sub_ops_[0]->getOpType() == "CrossEntropyLoss" && sub_ops_[1]->getOpType() == "Softmax")
		{
			opInd[0] = 1;
			opInd[1] = 0;
			opFlag = true;
		}
		if (opFlag == false)
		{
			DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::forward ]: the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
			return;
		}

		sub_ops_[opInd[0]]->forward(prev, next);
		sub_ops_[opInd[1]]->forward(next, next);
		//printf("finish SoftmaxCrossEntropyLossHOp\n");
	}

	template <typename Dtype>
	void SoftmaxCrossEntropyLossHOp<Dtype>::backward(const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev, const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
		const std::vector<std::shared_ptr<Tensor<Dtype>>> &prevDiff, const std::vector<std::shared_ptr<Tensor<Dtype>>> &nextDiff)
	{
		int opInd[2] = { -1 };
		bool opFlag = false;
		if (sub_ops_.size() != 2)
		{
			DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::forward ]:  sub_ops_.size() != 2 \n");
			return;
		}
		if (sub_ops_[0]->getOpType() == "Softmax" && sub_ops_[1]->getOpType() == "CrossEntropyLoss")
		{
			opInd[0] = 0;
			opInd[1] = 1;
			opFlag = true;
		}
		else if (sub_ops_[0]->getOpType() == "CrossEntropyLoss" && sub_ops_[1]->getOpType() == "Softmax")
		{
			opInd[0] = 1;
			opInd[1] = 0;
			opFlag = true;
		}
		if (opFlag == false)
		{
			DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::forward ]:  the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
			return;
		}
		sub_ops_[opInd[1]]->backward(prev, next, nextDiff, nextDiff);	//lastOutput in next[0], lastDiff will be generated in nextDiff
		sub_ops_[opInd[0]]->backward(prev, next, prevDiff, nextDiff);
		
	}

	INSTANTIATE_CLASS(SoftmaxCrossEntropyLossHOp);

}//namespace
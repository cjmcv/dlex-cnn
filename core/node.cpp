////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Basic operation unit
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <sstream>
#include "node.h"

namespace dlex_cnn
{
	template <typename Dtype>
	Node<Dtype>::Node()
	{
		phase_ = tind::Train;

		inputs_index_.clear();	//是否需要？
		outputs_index_.clear();
	}

	template <typename Dtype>
	Node<Dtype>::~Node()
	{

	}
	template <typename Dtype>
	int Node<Dtype>::hybridOpMap(std::string &inteOpType)
	{
		int opSize = sub_ops_.size();
		if (opSize <= 0)
			return -1;
		if (opSize == 2)
		{
			std::string opType0 = sub_ops_[0]->getOpType();
			std::string opType1 = sub_ops_[1]->getOpType();
			for (int i = 0; i < OP_DOUBLE_NUM; i++)
			{
				if ((opType0 == opListDouble[i][1] && opType1 == opListDouble[i][2]) ||
					(opType1 == opListDouble[i][1] && opType0 == opListDouble[i][2]))
				{
					inteOpType = opListDouble[i][0];
				}
			}
		}
		else if (opSize == 3)
		{
			//fill
		}
		else
		{
			DLOG_ERR("[ Node::hybridOpMap ]: sub_ops_.size() >= 4 that has not been implemented.\n");
			return -1;
		}

		return 0;
	}
	template <typename Dtype>
	int Node<Dtype>::inferInteOp()
	{
		if (sub_ops_.size() <= 0)
		{
			DLOG_ERR("[ Node::inferInteOp ]: sub_ops_.size() <= 0.\n");
			return -1;
		}
		if (sub_ops_.size() == 1)
		{	
			// 需要补充直接就是hop的情况
			inte_ops_ = sub_ops_[0];
			if (inte_ops_->getOpCategory() == tind::eHybridOp)
			{

			}
		}
		else
		{
			std::string inteOpStr;
			hybridOpMap(inteOpStr);
			printf("inteOp = %s\n", inteOpStr.c_str());
			
			int sIndex = -1;
			for (int i = 0; i < HOP_PHASEMAP_NUM; i++)
			{
				if (sIndex != -1)
					break;
				if (inteOpStr == hopPhaseMap[i][0])
					sIndex = i;
			}
			if (sIndex == -1)
			{
				DLOG_ERR("[ Node::inferInteOp ]: Can not find the hop with name < %s > in hopPhaseMap.\n", inteOpStr);
				return -1;
			}
			//printf("inte_ops = %s\n", hopPhaseMap[sIndex][phase_ + 1].c_str());
			inte_ops_ = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType(hopPhaseMap[sIndex][phase_+1].c_str());
			if (inte_ops_->getOpCategory() == tind::eHybridOp)
			{
				//printf("into inte_ops_->getOpCategory() == tind::eHybridOp\n");
				dynamic_cast<dlex_cnn::HybridOp<Dtype> *>(inte_ops_.get())->setSubOp(sub_ops_);
			}

			//dynamic_cast<dlex_cnn::InnerProductOp<float> *>(fc2.get())->setOpParam(innerProductParam2);
		}

		return 0;
	}

	template <typename Dtype>
	int Node<Dtype>::resetDataSize(int index, const std::vector<int> &shape)
	{
		input_shape_ = shape;
		int ret = inferOutShape();
		if (ret == 0)
			cpu_data_[index].reset(new Tensor<Dtype>(shape));
		return ret;
	};
	
	template <typename Dtype>
	int Node<Dtype>::writeNode2Text(FILE *fp)	//移至graph
	{
		////name, in_idx_count, in_idx（save name）
		//std::stringstream ss;
		//ss << name_ << "," << index_;
		//for (int i = 0; i < inputs_index_.size(); i++)
		//	ss << "," << inputs_index_[i];
		//ss << ";" << std::endl;

		//std::string opParam = getOpParamBufStr();
		//ss << opParam;

		//fprintf(fp, "%s\n", ss.str().c_str());

		return 0;
	}

	// no fixed
	template <typename Dtype>
	int Node<Dtype>::writeNode2Bin(FILE *fp)
	{
		//name, index, in_idx_count, in_idx
		int nameLen = name_.length();
		fwrite(&nameLen, sizeof(int), 1, fp);
		fwrite(name_.c_str(), sizeof(char), nameLen, fp);
		fwrite(&index_, sizeof(int), 1, fp);

		int inIdxSize = inputs_index_.size();
		fwrite(&inIdxSize, sizeof(int), 1, fp);
		for (int i = 0; i < inIdxSize; i++)
			fwrite(&inputs_index_[i], sizeof(int), 1, fp);

		std::string opParam = getOpParamBufStr();
		int opParamLen = opParam.length();
		fwrite(&opParamLen, sizeof(int), 1, fp);
		fwrite(opParam.c_str(), sizeof(char), opParamLen, fp);

		return 0;
	}

	template <typename Dtype>
	int Node<Dtype>::writeWB2Bin(FILE *fp)
	{
		//// use node name and index to verify
		//int nameLen = name_.length();
		//fwrite(&nameLen, sizeof(int), 1, fp);
		//fwrite(name_.c_str(), sizeof(char), nameLen, fp);
		//fwrite(&index_, sizeof(int), 1, fp);

		//int size = (cpu_data_.size() - 1) > 0 ? (cpu_data_.size() - 1) : 0;
		//fwrite(&size, sizeof(int), 1, fp);
		//
		//if (size == 0)
		//	return 0;

		//for (int i = 0; i < size; i++)
		//{
		//	int len = cpu_data_[i + 1]->getSize()[tind::e4D];
		//	fwrite(&len, sizeof(int), 1, fp);
		//	fwrite(cpu_data_[i + 1]->getData(), sizeof(Dtype), len, fp);
		//}
		return 0;
	}

	template <typename Dtype>
	int Node<Dtype>::readBin2WB(FILE *fp)	//移至graph
	{
		//// use node name and index to verify
		//int nameLen = name_.length();
		//fwrite(&nameLen, sizeof(int), 1, fp);
		//fwrite(name_.c_str(), sizeof(char), nameLen, fp);
		//fwrite(&index_, sizeof(int), 1, fp);

		//int size = (cpu_data_.size() - 1) > 0 ? (cpu_data_.size() - 1) : 0;
		//fwrite(&size, sizeof(int), 1, fp);

		//if (size == 0)
		//	return 0;

		//for (int i = 0; i < size; i++)
		//{
		//	int len = cpu_data_[i + 1]->getSize()[tind::e4D];
		//	fwrite(&len, sizeof(int), 1, fp);
		//	fwrite(cpu_data_[i + 1]->getData(), sizeof(Dtype), len, fp);
		//}
		return 0;
	}

	template <typename Dtype>
	void Node<Dtype>::serializeFromString(const std::string content)
	{
		//std::string layerType;
		//int number = 1;
		//int channels = 0;
		//int width = 0;
		//int height = 0;
		//std::stringstream ss(content);
		//ss >> name_ >> layerType >> channels >> height >> width;
		//if (layerType != sub_ops_[0]->getOpType())	//先push_back OP
		//{
		//	printf("layer type is invalidate.\n");
		//	return;
		//}

		//input_shape_.push_back(number);
		//input_shape_.push_back(channels);
		//input_shape_.push_back(height);
		//input_shape_.push_back(width);
		//sub_ops_[0]->inferOutShape(input_shape_, output_shape_);
	}
	INSTANTIATE_CLASS(Node);
}
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Create network
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "network_creator.h"

namespace dlex_cnn
{
	//////////////// Input //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createInputNode(std::string nodeName, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> input_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Input");
		if (input_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::InputOp<Dtype> *>(input_s.get())->setOpParam(param);

		std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > input;
		input.push_back(input_s);

		network.addNode(nodeName, input);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createInputNode(std::string nodeName, InputOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> input_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Input");
		if (input_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::InputOp<Dtype> *>(input_s.get())->setOpParam(param);

		std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > input;
		input.push_back(input_s);

		network.addNode(nodeName, input);
		return 0;
	}

	//////////////// Inner Product //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createInnerProductNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> fc_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("InnerProduct");
		if (fc_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::InnerProductOp<Dtype> *>(fc_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> fc;
		fc.push_back(fc_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, fc, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createInnerProductNode(std::string inNode, std::string name, InnerProductOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> fc_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("InnerProduct");
		if (fc_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::InnerProductOp<Dtype> *>(fc_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> fc;
		fc.push_back(fc_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, fc, inNodeNames);
		return 0;
	}

	//////////////// Convolution //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createConvNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> conv_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Convolution");
		if (conv_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::ConvolutionOp<Dtype> *>(conv_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> conv;
		conv.push_back(conv_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, conv, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createConvNode(std::string inNode, std::string name, ConvolutionOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> conv_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Convolution");
		if (conv_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::ConvolutionOp<Dtype> *>(conv_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> conv;
		conv.push_back(conv_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, conv, inNodeNames);
		return 0;
	}

	//////////////// Activation //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createActivationNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> act_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Activation");
		if (act_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::ActivationOp<Dtype> *>(act_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> act;
		act.push_back(act_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, act, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createActivationNode(std::string inNode, std::string name, ActivationOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> act_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Activation");
		if (act_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::ActivationOp<Dtype> *>(act_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> act;
		act.push_back(act_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, act, inNodeNames);
		return 0;
	}

	//////////////// Pooling //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createPoolNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> pool_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Pooling");
		if (pool_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::PoolingOp<Dtype> *>(pool_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> pool;
		pool.push_back(pool_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, pool, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createPoolNode(std::string inNode, std::string name, PoolingOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> pool_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Pooling");
		if (pool_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::PoolingOp<Dtype> *>(pool_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> pool;
		pool.push_back(pool_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, pool, inNodeNames);
		return 0;
	}

	//////////////// Softmax Cross Entropy Loss //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createSoftmaxLossNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("SoftmaxCrossEntropyLossH");
		if (sm_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::SoftmaxCrossEntropyLossHOp<Dtype> *>(sm_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> sm;
		sm.push_back(sm_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, sm, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createSoftmaxLossNode(std::string inNode, std::string name, SoftmaxCrossEntropyLossHOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("SoftmaxCrossEntropyLossH");
		if (sm_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::SoftmaxCrossEntropyLossHOp<Dtype> *>(sm_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> sm;
		sm.push_back(sm_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, sm, inNodeNames);
		return 0;
	}

	//////////////// Output //////////////////
	template <typename Dtype>
	int NetCreator<Dtype>::createOutputNode(std::string inNode, std::string name, std::string param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> out_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Output");
		if (out_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::OutputOp<Dtype> *>(out_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> out;
		out.push_back(out_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, out, inNodeNames);
		return 0;
	}
	template <typename Dtype>
	int NetCreator<Dtype>::createOutputNode(std::string inNode, std::string name, OutputOpParam param, NetWork<Dtype> &network)
	{
		std::shared_ptr<dlex_cnn::Op<Dtype>> out_s = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType("Output");
		if (out_s == NULL)
			return 1;

		dynamic_cast<dlex_cnn::OutputOp<Dtype> *>(out_s.get())->setOpParam(param);

		std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> out;
		out.push_back(out_s);

		std::vector<std::string> inNodeNames;
		inNodeNames.push_back(inNode);

		network.addNode(name, out, inNodeNames);
		return 0;
	}
	INSTANTIATE_CLASS(NetCreator);
}
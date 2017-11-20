////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  The Integration of each module about a model's
//          training and testing
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
//configure
#include "configure.h"

//network
#include "network.h"

#include "util/timer.h"

namespace dlex_cnn {

	template <typename Dtype>
	NetWork<Dtype>::NetWork()
	{
		
		printf("NetWork constructed.\n");
	}
	template <typename Dtype>
	NetWork<Dtype>::~NetWork()
	{
		printf("NetWork destructed.\n");
	}
	template <typename Dtype>
	int NetWork<Dtype>::netWorkInit(std::string name)
	{
		graph_.reset(new Graph<Dtype>());
		name_ = name;

		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::forward(const std::shared_ptr<Tensor<Dtype>> inputDataTensor, const std::shared_ptr<Tensor<Dtype>> labelDataTensor = NULL)
	{
		std::vector<std::shared_ptr<Tensor<Dtype>>> inputData;
		inputData.push_back(inputDataTensor);
		std::vector<std::string> nodeNames;
		nodeNames.push_back("input");
		graph_->setInNode(inputData, nodeNames);

		if (labelDataTensor != NULL)
		{
			std::vector<std::shared_ptr<Tensor<Dtype>>> labelData;
			labelData.push_back(labelDataTensor);
			std::vector<std::string> nodeNames2;
			nodeNames2.push_back("output");
			graph_->setOutNode(labelData, nodeNames2);
			//printf("finish set outnode\n");
		}

		graph_->forwardGraph();
		//printf("finish forward\n");
		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::backward()
	{
		graph_->backwardGraph();

		//update parameters
		const std::vector<std::shared_ptr<Node<Dtype>>> &nodes = graph_->getGraphNodes();
		for (int i = 0; i < nodes.size(); i++)
		{
			std::string opType = nodes[i]->getInteOp()->getOpType();
			if (!(opType == "Input" || opType == "Output"))
				optimizer_->update(nodes[i]);
		}
		return 0;
	}                                                                                                            
	//////////////////////////////////////////////////////////////////////////
	//test only!

	//train phase may use this
	template <typename Dtype>
	std::shared_ptr<Tensor<Dtype>> NetWork<Dtype>::testBatch(const std::shared_ptr<Tensor<Dtype>> inputDataTensor, const std::shared_ptr<Tensor<Dtype>> labelDataTensor = NULL)
	{
		////setPhase(Phase::Test);
		forward(inputDataTensor, labelDataTensor);
		//return lastOutput_[0];
		return NULL;
	}
	//////////////////////////////////////////////////////////////////////////
	//train only!
	template <typename Dtype>
	void NetWork<Dtype>::setOptimizer(std::shared_ptr<Optimizer<Dtype>> optimizer)
	{
		this->optimizer_ = optimizer;
	}
	template <typename Dtype>
	void NetWork<Dtype>::setLearningRate(const float lr)
	{
		this->optimizer_->setLearningRate(lr);
	}
	template <typename Dtype>
	void NetWork<Dtype>::addNode(std::string &nodeName, 
		std::vector<std::shared_ptr<Op<Dtype>>> &op, 
		std::vector<std::string> &inNodeNames = std::vector<std::string>())
	{
		graph_->addNode(nodeName, op, inNodeNames);
	}
	template <typename Dtype>
	int NetWork<Dtype>::switchPhase(int phase)
	{
		this->phase_ = phase;
		//graph_->phase_ = phase;
		graph_->setPhase(phase);
		const std::vector<std::shared_ptr<Node<Dtype>>> &nodes = graph_->getGraphNodes();
		for (int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->setPhase(phase);
			nodes[i]->inferInteOp();	// get new op
			nodes[i]->inferOutShape();
			nodes[i]->initOp();

			////graph_->nodes_[i]->initNode();	// 这里会改变node的权重
		}
		return 0;
	}

	template <typename Dtype>
	float NetWork<Dtype>::trainBatch(const std::shared_ptr<Tensor<Dtype>> inputDataTensor,
		const std::shared_ptr<Tensor<Dtype>> labelDataTensor)
	{
		//setPhase(Phase::Train);

		printf("inputDataTensor[0] = %d\n", inputDataTensor->getShape()[0]);
		//printf("trainBatch start forward\n");
		forward(inputDataTensor, labelDataTensor);
		//printf("trainBatch finish forward\n");

		Dtype loss = 100.0;
		graph_->getLoss("output", loss);

		//printf("trainBatch start backward\n");
		backward();
		//printf("trainBatch finish backward\n");
		return loss;
	}

	template <typename Dtype>
	int NetWork<Dtype>::getNodeData(const std::string &nodeName, std::shared_ptr<Tensor<Dtype>> &cpuData)
	{
		graph_->getNodeData(nodeName, cpuData);
		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::saveBinModel(const std::string& modelFile)
	{

		return true;
	}
	template <typename Dtype>
	int NetWork<Dtype>::loadBinModel(const std::string& modelFile)
	{

		return true;
	}
	template <typename Dtype>
	int NetWork<Dtype>::saveStageModel(const std::string &path, const int stage)
	{
		std::string structFileName = "iter_" + std::to_string(stage) + ".struct";
		std::string paramFileName = "iter_" + std::to_string(stage) + ".param";

		FILE *stFp = fopen(structFileName.c_str(), "w");
		graph_->writeGraph2Text(stFp);

		std::stringstream optss;
		optss << "optimizer:" << optimizer_->getOptName() << ",lr:" << optimizer_->getLearningRate() << ";";
		fprintf(stFp, "%s\n", optss.str().c_str());
		
		fclose(stFp);

		FILE *paramFp = fopen(paramFileName.c_str(), "wb");
		graph_->writeGraphParam2Bin(paramFp);
		fclose(paramFp);

		return 0;
	}

	template <typename Dtype>
	int NetWork<Dtype>::readHyperParams(FILE *fp)
	{
		char cc[1000];
		while (EOF != fscanf(fp, "%s", cc))	// Fetch optimizer's parameters
		{
			std::string cstr(cc);
			printf("read3: %s\n", cstr.c_str());

			std::string optStr = fetchSubStr(cstr, "optimizer:", ",");
			float lr = atof(fetchSubStr(cstr, "lr:", ";").c_str());

			std::shared_ptr<dlex_cnn::Optimizer<Dtype>> optimizer;
			if (dlex_cnn::Optimizer<Dtype>::getOptimizerByStr(optStr, optimizer))
			{
				DLOG_ERR("[ NetWork::readHyperParams ]: Can not find optimizer by name - %s \n", optStr.c_str());
				return -1;
			}
			this->setOptimizer(optimizer);

			if (lr > 0)
				this->setLearningRate(lr);
			else
			{
				DLOG_ERR("[ NetWork::readHyperParams ]: Invalid learning rate -> () \n", lr);
				return -1;
			}

			printf("read22_0: %s, %f\n", optStr.c_str(), lr);
		}
		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::loadStageModel(const std::string &path, const int stage)
	{
		//readText2Graph(FILE *fp);
		std::string structFileName = "iter_" + std::to_string(stage) + ".struct";
		std::string paramFileName = "iter_" + std::to_string(stage) + ".param";

		FILE *stFp = fopen(structFileName.c_str(), "r");
		graph_->readText2Graph(stFp);
		readHyperParams(stFp);

		fclose(stFp);

		FILE *paramFp = fopen(paramFileName.c_str(), "rb");
		graph_->readBin2GraphParam(paramFp);
		fclose(paramFp);

		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::netWorkShow()
	{
		printf("***************************************************** \n");
		printf("**************  Network's name: <%s>. *************\n", name_.c_str());
		printf("======================= Graph ======================= \n");
		graph_->graphShow();
		printf(">>>>>>>>>>>>>>>>>>>>> Optimizer <<<<<<<<<<<<<<<<<<<<< \n");
		printf("lr: %f\n", optimizer_->getLearningRate());
		printf("***************************************************** \n");
		return 0;
	}
	INSTANTIATE_CLASS(NetWork);

}//namespace
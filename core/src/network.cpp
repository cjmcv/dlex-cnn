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
	int NetWork<Dtype>::netWorkInit(std::string name, tind::Mode device_mode)
	{
		graph_.reset(new Graph<Dtype>());
		name_ = name;
		device_mode_ = device_mode;

		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::forward(const std::shared_ptr<Tensor<Dtype>> input_data_tensor, const std::shared_ptr<Tensor<Dtype>> label_data_tensor)
	{
		std::vector<std::shared_ptr<Tensor<Dtype>>> input_data;
		input_data.push_back(input_data_tensor);
		std::vector<std::string> node_names;
		node_names.push_back("input");
		graph_->setInNode(input_data, node_names);

		if (label_data_tensor != NULL)
		{
			std::vector<std::shared_ptr<Tensor<Dtype>>> label_data;
			label_data.push_back(label_data_tensor);
			std::vector<std::string> node_names2;
			node_names2.push_back("output");
			graph_->setOutNode(label_data, node_names2);
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
			std::string op_type = nodes[i]->getInteOp()->getOpType();
			if (!(op_type == "Input" || op_type == "Output"))
				optimizer_->update(nodes[i]);
		}
		return 0;
	}                                                                                                            
	//////////////////////////////////////////////////////////////////////////
	//test only!

	//train phase may use this
	template <typename Dtype>
	std::shared_ptr<Tensor<Dtype>> NetWork<Dtype>::testBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor, const std::shared_ptr<Tensor<Dtype>> label_data_tensor)
	{
		////setPhase(Phase::Test);
		forward(input_data_tensor, label_data_tensor);
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
	void NetWork<Dtype>::addNode(const std::string &node_name, 
		const std::vector<std::shared_ptr<Op<Dtype>>> &op, 
		const std::vector<std::string> &in_node_names)
	{
		graph_->addNode(node_name, op, in_node_names);
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
	float NetWork<Dtype>::trainBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor,
		const std::shared_ptr<Tensor<Dtype>> label_data_tensor)
	{
		//setPhase(Phase::Train);

		printf("input_data_tensor[0] = %d\n", input_data_tensor->getShape()[0]);
		//printf("trainBatch start forward\n");
		forward(input_data_tensor, label_data_tensor);
		//printf("trainBatch finish forward\n");

		Dtype loss = 100.0;
		graph_->getLoss("output", loss);

		//printf("trainBatch start backward\n");
		backward();
		//printf("trainBatch finish backward\n");
		return loss;
	}

	template <typename Dtype>
	int NetWork<Dtype>::getNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &cpu_data)
	{
		graph_->getNodeData(node_name, cpu_data);
		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::saveBinModel(const std::string& model_file)
	{

		return true;
	}
	template <typename Dtype>
	int NetWork<Dtype>::loadBinModel(const std::string& model_file)
	{

		return true;
	}
	template <typename Dtype>
	int NetWork<Dtype>::saveStageModel(const std::string &path, const int stage)
	{
		std::string struct_file_name = "iter_" + std::to_string(stage) + ".struct";
		std::string param_file_name = "iter_" + std::to_string(stage) + ".param";

		FILE *st_fp = fopen(struct_file_name.c_str(), "w");
		graph_->writeGraph2Text(st_fp);

		std::stringstream optss;
		optss << "optimizer:" << optimizer_->getOptName() << ",lr:" << optimizer_->getLearningRate() << ";";
		fprintf(st_fp, "%s\n", optss.str().c_str());
		
		fclose(st_fp);

		FILE *param_fp = fopen(param_file_name.c_str(), "wb");
		graph_->writeGraphParam2Bin(param_fp);
		fclose(param_fp);

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

			std::string opt_str = fetchSubStr(cstr, "optimizer:", ",");
			float lr = atof(fetchSubStr(cstr, "lr:", ";").c_str());

			std::shared_ptr<dlex_cnn::Optimizer<Dtype>> optimizer;
			if (dlex_cnn::Optimizer<Dtype>::getOptimizerByStr(opt_str, optimizer))
			{
				DLOG_ERR("[ NetWork::readHyperParams ]: Can not find optimizer by name - %s \n", opt_str.c_str());
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

			printf("read22_0: %s, %f\n", opt_str.c_str(), lr);
		}
		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::loadStageModel(const std::string &path, const int stage)
	{
		//readText2Graph(FILE *fp);
		std::string struct_file_name = "iter_" + std::to_string(stage) + ".struct";
		std::string param_file_name = "iter_" + std::to_string(stage) + ".param";

		FILE *st_fp = fopen(struct_file_name.c_str(), "r");
		graph_->readText2Graph(st_fp);
		readHyperParams(st_fp);

		fclose(st_fp);

		FILE *param_fp = fopen(param_file_name.c_str(), "rb");
		graph_->readBin2GraphParam(param_fp);
		fclose(param_fp);

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

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

#include "network.h"
#include "util/timer.h"

namespace dlex_cnn {

	template <typename Dtype>
	NetWork<Dtype>::NetWork(std::string name)
	{
		name_ = name;
		graph_.reset(new Graph<Dtype>());

		DLOG_INFO("NetWork constructed.");
	}
	template <typename Dtype>
	NetWork<Dtype>::~NetWork()
	{
		DLOG_INFO("NetWork destructed.");
	}
	template <typename Dtype>
	int NetWork<Dtype>::netParamsInit()
	{
		graph_->paramsInit();

		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::feedDataByPrefetcher()
	{
		int ret = 0;
		std::pair < std::shared_ptr<Tensor<Dtype>>, std::shared_ptr<Tensor<Dtype>> > *batch;
		prefetcher_.feedBatchOut(&batch);

		input_data_vec_.clear();
		input_data_vec_.push_back(batch->first);
		ret = graph_->setInNode(input_data_vec_);

		label_data_vec_.clear();
		label_data_vec_.push_back(batch->second);
		ret += graph_->setOutNode(label_data_vec_);

#ifdef USE_CUDA
		// Here is processed by the main thread, which is not the prefetcher one.
		// Ensure the copy is synchronous, so that the next batch is not copied in meanwhile.
		if (Task::mode() == tind::GPU)
			CUDA_DCHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
		prefetcher_.refillBuffer(&batch);

		if (ret != 0)
			return -1;

		return 0;
	}
	template <typename Dtype>
	int NetWork<Dtype>::forward(const std::shared_ptr<Tensor<Dtype>> input_data_tensor, const std::shared_ptr<Tensor<Dtype>> label_data_tensor)
	{
		if (input_data_tensor == NULL)
		{
			int ret = feedDataByPrefetcher();
			if (ret != 0)
				return -1;
		}
		else
		{
			int ret = 0;
			if (Task::mode() == tind::GPU)
			{
#ifdef USE_CUDA
				input_data_tensor->checkPushGpuData();
#else
				DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif	
			}
			input_data_vec_.clear();
			input_data_vec_.push_back(input_data_tensor);
			ret = graph_->setInNode(input_data_vec_);

			if (label_data_tensor != NULL)
			{
				if (Task::mode() == tind::GPU)
				{
#ifdef USE_CUDA
					label_data_tensor->checkPushGpuData();
#else
					DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif				
				}

				label_data_vec_.clear();
				label_data_vec_.push_back(label_data_tensor);
				ret += graph_->setOutNode(label_data_vec_);
			}
			if (ret != 0)
				return -1;
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
			{
				if (Task::mode() == tind::CPU)
					optimizer_->update(nodes[i]);
				else
				{
#ifdef USE_CUDA
					optimizer_->update_gpu(nodes[i]);
#else
					DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif
				}
					
			}
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
	inline int NetWork<Dtype>::setIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names)
	{
		return graph_->setIONodeName(in_node_names, out_node_names);
	}

	template <typename Dtype>
	float NetWork<Dtype>::trainBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor,
		const std::shared_ptr<Tensor<Dtype>> label_data_tensor)
	{
		//setPhase(Phase::Train);

		//printf("input_data_tensor[0] = %d\n", input_data_tensor->getShape()[0]);
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
	int NetWork<Dtype>::getNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &data)
	{
		graph_->getNodeData(node_name, data);
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
				DLOG_ERR("[ NetWork::readHyperParams ]: Can not find optimizer by name - %s.", opt_str.c_str());
				return -1;
			}
			this->setOptimizer(optimizer);

			if (lr > 0)
				this->setLearningRate(lr);
			else
			{
				DLOG_ERR("[ NetWork::readHyperParams ]: Invalid learning rate -> ().", lr);
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
		DLOG_INFO("***************************************************** ");
		DLOG_INFO("**************  Network's name: <%s>. *************\n", name_.c_str());
		DLOG_INFO("======================= Graph ======================= ");
		graph_->graphShow();
		DLOG_INFO(">>>>>>>>>>>>>>>>>>>>> Optimizer <<<<<<<<<<<<<<<<<<<<< ");
		DLOG_INFO("lr: %f\n", optimizer_->getLearningRate());
		DLOG_INFO("***************************************************** ");
		return 0;
	}
	INSTANTIATE_CLASS(NetWork);

}//namespace

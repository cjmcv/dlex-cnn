////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node graph, mainly contains nodes, 
//          support for node operation including forward and backward
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <sstream>
#include "graph.h"
#include "task.h"
#include "util/math_functions.h"

namespace dlex_cnn
{
	template <typename Dtype>
	Graph<Dtype>::Graph()
	{

	}

	template <typename Dtype>
	Graph<Dtype>::~Graph()
	{

	}

	template <typename Dtype>
	int Graph<Dtype>::getNodeIndex(const std::string &node_name, int &index)
	{
		std::map<std::string, int>::iterator it = nodes_index_map_.find(node_name);
		if (it == nodes_index_map_.end())
			return -1;

		index = it->second;
		return 0;
	}

	template <typename Dtype>
	void Graph<Dtype>::addNode(const std::string &node_name,
		const std::vector< std::shared_ptr< Op<Dtype> > > &op,
		const std::vector<std::string> &in_node_names)
	{
		std::shared_ptr<Node<Dtype>> node = std::make_shared<Node<Dtype>>();
		
		const int node_idx = nodes_.size();
		node->setIndex(node_idx);
		node->setName(node_name);
		nodes_index_map_[node_name] = node_idx;

		for (int i = 0; i < op.size(); i++)
			node->addSubOps(op[i]);

		node->setPhase(phase_);
		node->inferInteOp();
		//node->input_shape_ = cpu_data_[cpu_data_.size() - 1]->getShape();

		const std::shared_ptr<Op<Dtype>> inteOp = node->getInteOp();
		if (inteOp->getOpType() == "Input")
		{
			in_nodes_map_[node_name] = node_idx;
		}
		else
		{
			for (int idx = 0; idx < in_node_names.size(); idx++)
			{
				int index = -1;
				int ret = getNodeIndex(in_node_names[idx], index);
				if (ret != 0)
				{
					DLOG_ERR("[ Graph::addNode ]: Can not get node with name %s \n", in_node_names[idx].c_str());
					continue;
				}
				node->addInputName(in_node_names[idx]); // Create link from current node to previous node
				node->addInputIdx(index);	
				nodes_[index]->addOutputName(node->getName()); // Create link from previous node to current node 
				nodes_[index]->addOutputIdx(node_idx);  
			}
			//node->input_shape_ = nodes_[node->inputs_index_[0]]->output_shape_;
			node->setInputShape(nodes_[node->getInputIdx()[0]]->getOutputShape());
			if (inteOp->getOpType() == "Output")
				out_nodes_map_[node_name] = node_idx;
		}
		node->inferOutShape();
		node->initOp();
		node->initNode();
		nodes_.push_back(node);
		DLOG_INFO("Add node: %s.", node->getInteOp()->getOpType().c_str());
	}

	// Initialize all of the weight and bias.
	template <typename Dtype>
	void Graph<Dtype>::paramsInit()
	{
		for (int i = 0; i < nodes_.size(); i++)
		{	
			const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[i]->getDataVec();

			if (Task::mode() == tind::CPU)
			{
				if (data_vec.size() > 1)
				{
					normal_distribution_init<Dtype>(data_vec[1]->getSize()[tind::e4D], 0.0f, 0.1f, (Dtype *)data_vec[1]->getCpuData());
					if (data_vec.size() > 2)
						dlex_set<Dtype>(data_vec[2]->getSize()[tind::e4D], 0.0f, (Dtype *)data_vec[2]->getCpuData());
				}
			}
			else
			{
#ifdef USE_CUDA
				if (data_vec.size() > 1)
				{
					dlex_gpu_rng_gaussian<Dtype>(data_vec[1]->getSize()[tind::e4D], 0.0f, 0.1f, (Dtype *)data_vec[1]->getGpuData());
					if (data_vec.size() > 2)
						dlex_gpu_set<Dtype>(data_vec[2]->getSize()[tind::e4D], 0.0f, (Dtype *)data_vec[2]->getGpuData());
				}
#else
				DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif
			}
		}
	}

	template <typename Dtype>
	int Graph<Dtype>::setIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names)
	{
		in_node_names_.assign(in_node_names.begin(), in_node_names.end());
		out_node_names_.assign(out_node_names.begin(), out_node_names.end());

		return 0;
	}
	// fill the input data in innode->cpu_data_[0]
	template <typename Dtype>
	int Graph<Dtype>::setInNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> input_data)
	{
		if (input_data.size() != in_node_names_.size())
		{ 
			DLOG_ERR("[ Graph::setInNode ]: input_data should have the same size with node_names\n");
			return -1;
		}
		if (input_data.size() <= 0)
		{
			DLOG_ERR("[ Graph::setInNode ]: input_data is invalid\n");
			return -1;
		}

		const int data_num = input_data[0]->getShape()[0];
		for (int i = 0; i < input_data.size(); i++)
		{
			if (data_num != input_data[i]->getShape()[0])
			{
				DLOG_ERR("[ Graph::setInNode ]: Each block of data should has the same num\n");
				return -1;
			}
		}

		for (int i = 0; i < input_data.size(); i++)
		{
			int index = -1;
			int ret = getNodeIndex(in_node_names_[i], index);
			if (ret != 0)
			{
				DLOG_ERR("[ Graph::setInNode ]: Can not get node with name %s \n", in_node_names_[i].c_str());
				return -1;
			}

			if (nodes_[index]->getDataVec()[0]->getSize()[tind::e4D] != input_data[i]->getSize()[tind::e4D])
			{
				nodes_[index]->setInputShape(input_data[i]->getShape());
				nodes_[index]->resetDataSize(0, input_data[i]->getShape());
				//nodes_[index]->inferOutShape();
				//cpu_data[0].reset(new Tensor<Dtype>(input_data[i]->getShape()));
			}
			if (Task::mode() == tind::CPU)
				input_data[i]->copyDataTo(*nodes_[index]->getDataVec()[0], tind::eHost2Host);
			else
				input_data[i]->copyDataTo(*nodes_[index]->getDataVec()[0], tind::eDevice2Device);
		}
		return 0;
	}

	//fill the label data in outnode->cpu_data_[1]
	template <typename Dtype>
	int Graph<Dtype>::setOutNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> label_data)
	{
		if (label_data.size() != out_node_names_.size())
		{
			DLOG_ERR("[ Graph::setOutNode ]: label_data should have the same size with node_names\n");
			return -1;
		}
		if (label_data.size() <= 0)
		{
			DLOG_ERR("[ Graph::setOutNode ]: label_data is invalid\n");
			return -1;
		}

		//printf("s0 set out node\n");
		for (int i = 0; i < label_data.size(); i++)
		{
			int index = -1;
			int ret = getNodeIndex(out_node_names_[i], index);
			//printf("s1 set out node\n");
			if (ret != 0)
			{
				DLOG_ERR("[ Graph::setOutNode ]: Can not get node with name %s. \n", out_node_names_[i].c_str());
				return -1;
			}
			
			// The format of output node is ( output[0], label[1], loss[2] )
			const int vec_size = nodes_[index]->getDataVec().size();
			if (vec_size != 3)
			{
				DLOG_ERR("[ Graph::setOutNode ]: Output node is not contains 3 tenosr. \n");
				return - 1;
			}
			/*if (nodes_[i]->inte_ops_->getOpDiff()[0]->getSize()[tind::e4D] != nodes_[i]->cpu_data_[0]->getSize()[tind::e4D])
				nodes_[i]->inte_ops_->getOpDiff()[0].reset(new Tensor<Dtype>(nodes_[i]->cpu_data_[0]->shape_));*/

			//printf("s set out node\n");
			if (nodes_[index]->getDataVec()[1]->getSize()[tind::e4D] != label_data[i]->getSize()[tind::e4D])
				nodes_[index]->resetDataSize(1, label_data[i]->getShape());

			const std::vector<std::shared_ptr<Tensor<Dtype>>> cpu_data = nodes_[index]->getDataVec();

			if (Task::mode() == tind::CPU)
				label_data[i]->copyDataTo(*cpu_data[1], tind::eHost2Host);
			else
				label_data[i]->copyDataTo(*cpu_data[1], tind::eDevice2Device);

			//printf("finish set out node\n");
		}
		return 0;
	}

	// forward graph by DFS
	template <typename Dtype>
	int Graph<Dtype>::forwardGraph()
	{
		//printf("NetWork forward begin.\n");
		if (nodes_.size() <= 1)
		{
			DLOG_ERR("[ Graph::forwardGraph ]: A graph should has more than 2 nodes. \n");
			return -1;
		}
		if (in_nodes_map_.size() <= 0 || out_nodes_map_.size() <= 0)
		{
			DLOG_ERR("[ Graph::forwardGraph ]: input node or output node is empty. \n");
			return -1;
		}

		while (!nodes_idx_stack_.empty()) 
			nodes_idx_stack_.pop();

		std::map<std::string, int>::iterator it = in_nodes_map_.begin();
		while (it != in_nodes_map_.end())	// push all of the input nodes
		{
			//printf("while (it != in_nodes_map_.end())\n");
			//it->first; it->second;
			nodes_idx_stack_.push(it->second);
			it++;
		}

		//// 注意 当前为取[0]号输出，后面支持多输出（兼容前向？或在计算图模型中处理？）
		while (!nodes_idx_stack_.empty())	//DFS , 后需添加depends
		{
			//printf("!nodes_idx_stack_.empty()\n");
			int idx = nodes_idx_stack_.top();
			nodes_idx_stack_.pop();

			const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->getInteOp();
			if (inte_op_idx->getOpType() == "Output")
				continue ;
			
			// recheck batch size
			const std::vector<int> idx_out_shape = nodes_[idx]->getOutputShape();
			const std::vector<int> output_idx  = nodes_[idx]->getOutputIdx();
			const std::vector<int> idx_next_in_shape = nodes_[output_idx[0]]->getInputShape();
			if (idx_out_shape != idx_next_in_shape)
			{
				nodes_[output_idx[0]]->setInputShape(idx_out_shape);
				//nodes_[output_idx[0]]->inferOutShape();
				nodes_[output_idx[0]]->resetDataSize(0, idx_out_shape);	// it will call inferOutShape
				//nodes_[output_idx[0]]->initOp();	// no, 前向和反向过程中，只有在用到时发现某块数据维度不对，才修改要用的那块内存大小。
			}

			// The bottom data of forward is saved in the node that executing forward operation. 
			if (Task::mode() == tind::CPU)
			{
				//nodes_[output_idx[0]]->getDataVec()[0]->setCpuZero();
				inte_op_idx->forward(nodes_[idx]->getDataVec(), nodes_[output_idx[0]]->getDataVec());
			}
			else
			{
				nodes_[output_idx[0]]->getDataVec()[0]->setGpuZero();
				inte_op_idx->forward_gpu(nodes_[idx]->getDataVec(), nodes_[output_idx[0]]->getDataVec());
			}

			//float *outdata0 = (float *)nodes_[idx]->getDataVec()[1]->getCpuData();
			//for (int j = 0; j < nodes_[idx]->getDataVec()[1]->getSize()[3]; j++)
			//	printf("%f,", outdata0[j]);
			//printf("\n");

			for (int i = 0; i < output_idx.size(); i++)
			{
				//printf("push %d\n", nodes_[idx]->outputs_index_[i]);
				nodes_idx_stack_.push(output_idx[i]);
			}
		}

		return 0;
	}
	
	// backward graph by DFS
	template <typename Dtype>
	int Graph<Dtype>::backwardGraph()
	{
		if (nodes_.size() <= 1)
		{
			DLOG_ERR("[ Graph::backwardGraph ]: A graph should has more than 2 nodes. \n");
			return -1;
		}
		if (in_nodes_map_.size() <= 0 || out_nodes_map_.size() <= 0)
		{
			DLOG_ERR("[ Graph::backwardGraph ]: input node or output node is empty. \n");
			return -1;
		}

		Timer timer;
		while (!nodes_idx_stack_.empty())
			nodes_idx_stack_.pop();

		std::map<std::string, int>::iterator it = out_nodes_map_.begin();
		while (it != out_nodes_map_.end())	// push all of the input node
		{
			//printf("while (it != out_nodes_map_.end())\n");
			//it->first; it->second;
			nodes_idx_stack_.push(it->second);
			it++;
		}

		while (!nodes_idx_stack_.empty())	//DFS , 后需添加depends
		{
			int idx = nodes_idx_stack_.top();
			//printf("backward idx = %d, input_idx = %d\n", idx, nodes_[idx]->inputs_index_.size());
			nodes_idx_stack_.pop();

			const std::vector<int> input_idx = nodes_[idx]->getInputIdx();
			for (int i = 0; i < input_idx.size(); i++)
			{
				//printf("push %d\n", nodes_[idx]->inputs_index_[i]);
				nodes_idx_stack_.push(input_idx[i]);
			}

			const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->getInteOp();
			const std::vector<std::shared_ptr<Tensor<Dtype>>> data_idx = nodes_[idx]->getDataVec();
		
			// recheck batch size : diff_
			if (inte_op_idx->getOpDiff()[0]->getSize()[tind::e4D] != data_idx[0]->getSize()[tind::e4D])
				inte_op_idx->getOpDiff()[0].reset(new Tensor<Dtype>(data_idx[0]->getShape()));

			if (inte_op_idx->getOpType() == "Output")
				continue;

			// The bottom data of backward is saved in the node that executing backward operation. 
			const std::vector<int> output_idx = nodes_[idx]->getOutputIdx();
			if (Task::mode() == tind::CPU)
			{
				//inte_op_idx->getOpDiff()[0]->setCpuZero();
				inte_op_idx->backward(data_idx, nodes_[output_idx[0]]->getDataVec(),
					inte_op_idx->getOpDiff(), nodes_[output_idx[0]]->getInteOp()->getOpDiff());
			}
			else
			{
				inte_op_idx->getOpDiff()[0]->setGpuZero();
				inte_op_idx->backward_gpu(data_idx, nodes_[output_idx[0]]->getDataVec(),
					inte_op_idx->getOpDiff(), nodes_[output_idx[0]]->getInteOp()->getOpDiff());
			}

		}

		//// 注意 当前为取[0]号输出，后面支持多输出（兼容前向？或在计算图模型中处理？）！！！

		//for (int i = (int)(nodes_.size()) - 2; i >= 0; i--)
		//{
		//	//printf("backward[%d].\n", i);
		//	timer.Start();
		//	nodes_[i]->inte_ops_->getOpDiff()[0]->setCpuZero();
		//	nodes_[i]->inte_ops_->backward(nodes_[i]->cpu_data_, nodes_[nodes_[i]->outputs_index_[0]]->cpu_data_,
		//		nodes_[i]->inte_ops_->getOpDiff(), nodes_[nodes_[i]->outputs_index_[0]]->inte_ops_->getOpDiff());
		//	float us = timer.MilliSeconds();
		//	printf("time[%d]: %f ms\n", i, us);
		//}
		return 0;//nodes_[nodes_.size() - 1]->cpu_data_[2];
	}

	template <typename Dtype>
	int Graph<Dtype>::getLoss(const std::string &node_name, Dtype &loss)
	{
		loss = 100.0;

		int index = -1;
		int ret = getNodeIndex(node_name, index);
		if (ret != 0)
		{
			DLOG_ERR("[ Graph::getLoss ]: Can not get node with name < %s >.\n", node_name.c_str());
			return -1;
		}
		if (nodes_[index]->getInteOp()->getOpType() != "Output")
		{
			DLOG_ERR("[ Graph::getLoss ]: The node with name < %s >, is not an output node.\n", node_name.c_str());
			return -1;
		}
		loss = *(Dtype *)(nodes_[index]->getDataVec()[2]->getCpuData());

		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::getNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &cpuData)
	{
		int index = -1;
		int ret = getNodeIndex(node_name, index);
		if (ret != 0)
		{
			DLOG_ERR("[ Graph::getNodeData ]: Can not get node with name < %s >.\n", node_name.c_str());
			return -1;
		}
		cpuData = nodes_[index]->getDataVec()[0];

		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::graphShow()
	{
		//printf("NetWork forward begin.\n");
		if (nodes_.size() <= 1)
		{
			DLOG_ERR("[ Graph::forwardGraph ]: A graph should has more than 2 nodes. \n");
			return -1;
		}

		while (!nodes_idx_stack_.empty())
			nodes_idx_stack_.pop();

		std::map<std::string, int>::iterator it = in_nodes_map_.begin();
		while (it != in_nodes_map_.end())	// push all of the input node
		{
			//it->first; it->second;
			nodes_idx_stack_.push(it->second);
			it++;
		}

		while (!nodes_idx_stack_.empty())	//DFS , 后需添加depends
		{
			int idx = nodes_idx_stack_.top();
			printf("================================================(%d)== \n", idx);
			nodes_idx_stack_.pop();
			
			const std::string cur_node_name = nodes_[idx]->getName();
			const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->getInteOp();
			printf("*  node name: <%s> , op type: <%s>.\n", cur_node_name.c_str(), inte_op_idx->getOpType().c_str());

			// weight / blas
			const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[idx]->getDataVec();
			const std::vector<int> data_shape = data_vec[0]->getShape();
			printf("*  data: (%d, %d, %d, %d). \n", data_shape[tind::eNum], data_shape[tind::eChannels], data_shape[tind::eHeight], data_shape[tind::eWidth]);
			if (data_vec.size() >= 2)
			{
				const std::vector<int> weight_shape = data_vec[1]->getShape();
				printf("*  weight: (%d, %d, %d, %d). \n", weight_shape[tind::eNum], weight_shape[tind::eChannels], weight_shape[tind::eHeight], weight_shape[tind::eWidth]);
				
				if (data_vec.size() >= 3)
				{
					const std::vector<int> blas_shape = data_vec[2]->getShape();
					printf("*  blas: (%d, %d, %d, %d). \n", blas_shape[tind::eNum], blas_shape[tind::eChannels], blas_shape[tind::eHeight], blas_shape[tind::eWidth]);
				}
				else
					printf("*  blas: None. \n");
			}
			else
			{
				printf("*  weight: None. \n");
				printf("*  blas: None. \n");
			}	

			// gradient / diff
			const std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_vec = inte_op_idx->getOpGradient();
			if (gradient_vec.size() != 0)
			{
				const std::vector<int> shape = gradient_vec[0]->getShape();
				printf("*  gradient: (%d, %d, %d, %d). \n", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}
			else
				printf("*  gradient: None. \n");

			const std::vector<std::shared_ptr<Tensor<Dtype>>> diff_vec = inte_op_idx->getOpDiff();
			if (diff_vec.size() != 0)
			{
				const std::vector<int> shape = diff_vec[0]->getShape();
				printf("*  diff: (%d, %d, %d, %d). \n", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}
			else
				printf("*  diff: None. \n");

			// input / output
			const std::vector<int> input_idx = nodes_[idx]->getInputIdx();
			const std::vector<int> output_idx = nodes_[idx]->getOutputIdx();
			for (int i = 0; i < input_idx.size(); i++)
			{
				const std::vector<int> shape = nodes_[input_idx[i]]->getOutputShape(); // getDataVec()[0]->getShape();
				printf("*  %s <%s> (%d, %d, %d, %d) -> %s. \n", nodes_[input_idx[i]]->getName().c_str(),
					nodes_[input_idx[i]]->getInteOp()->getOpType().c_str(),
					shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth],
					cur_node_name.c_str());
			}
			for (int i = 0; i < output_idx.size(); i++)
			{
				const std::vector<int> shape = nodes_[output_idx[i]]->getInputShape(); // getDataVec()[0]->getShape();
				printf("*  %s -> %s <%s> (%d, %d, %d, %d). \n", cur_node_name.c_str(),
					nodes_[output_idx[i]]->getName().c_str(),
					nodes_[output_idx[i]]->getInteOp()->getOpType().c_str(),
					shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}

			for (int i = 0; i < output_idx.size(); i++)
				nodes_idx_stack_.push(output_idx[i]);
		}
		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::writeGraph2Text(FILE *fp)
	{
		std::stringstream optss;
		optss << "nodes_size:" << nodes_.size() << ";";
		fprintf(fp, "%s\n", optss.str().c_str());

		for (int i = 0; i < nodes_.size(); i++)
		{
			//nodes_[i]->writeNode2Text(fp);
			////name, in_idx（save name）
			std::stringstream ss;
			ss << "node_name:" << nodes_[i]->getName() << ",op_type:" << nodes_[i]->getInteOp()->getOpType() << "," << "inNodesName:";

			const std::vector<std::string> input_names = nodes_[i]->getInputName();
			ss << "(";
			for (int j = 0; j < input_names.size(); j++)
			{
				ss << input_names[j];
				if (j < input_names.size() - 1)
					ss << ",";
			}
			ss << ");" << std::endl;

			std::string op_param = nodes_[i]->getOpParamBufStr();
			ss << op_param;

			fprintf(fp, "%s\n", ss.str().c_str());
		}
		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::readText2Graph(FILE *fp)
	{
		nodes_.clear();

		char cc[1000];

		int graph_size = 0;
		if (EOF != fscanf(fp, "%s", cc))	// Fetch the first line to get the graph size.
		{
			std::string cstr(cc);
			printf("read0: %s\n", cstr.c_str());
			graph_size = atoi(fetchSubStr(cstr, "nodes_size:", ";").c_str());
			printf("read0: %d\n", graph_size);
		}
		std::string node_name, op_type;
		std::vector<std::string> in_node_names;
		int line_count = 0;
		while (line_count < graph_size * 2)
		{
			if (EOF == fscanf(fp, "%s", cc))
				return -1;
			line_count++;

			// Fetch each node's parameters
			std::string cstr(cc);
			if (line_count % 2 == 1)          // Current and input nodes' name 
			{
				printf("read1: %s\n", cstr.c_str());

				// Fetch node name
				node_name = fetchSubStr(cstr, "node_name:", ",");

				// Fetch operator's name in this node
				op_type = fetchSubStr(cstr, "op_type:", ",");
				printf("read11_0: %s, %s\n", node_name.c_str(), op_type.c_str());

				if (op_type == "Input")
					continue;

				// Fetch input nodes' name of this node
				std::string inNamesStr = fetchSubStr(cstr, "inNodesName:(", ")");
				int comma_flag;
				in_node_names.clear();
				while ((comma_flag = inNamesStr.find(",")) != -1)
				{
					std::string name = inNamesStr.substr(0, comma_flag);
					in_node_names.push_back(name);
					inNamesStr = inNamesStr.substr(comma_flag + 1, inNamesStr.length());
				}
				in_node_names.push_back(inNamesStr);
				for (int i = 0; i < in_node_names.size(); i++)
					printf("in_node_names[%d]: %s\n", i, in_node_names[i].c_str());

			}
			else
			{
				// OpParams have been saved in this line
				printf("read2: %s\n", cstr.c_str());

				// Create node here according to previous information.
				std::shared_ptr<dlex_cnn::Op<Dtype>> node_op = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType(op_type);
				if (node_op == NULL)
					DLOG_ERR("[ Graph<Dtype>::readText2Graph ]: Can not create Op by type (%s) \n", op_type.c_str());
				node_op->setOpParam(cstr);
				std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > node_ops;
				node_ops.push_back(node_op);

				this->addNode(node_name, node_ops, in_node_names);
			}

		}
		//std::stringstream optss;
		//optss << "nodes_size:" << nodes_.size() << ";";
		//fprintf(fp, "%s\n", optss.str().c_str());

		//for (int i = 0; i < nodes_.size(); i++)
		//{
		//	nodes_[i]->writeNode2Text(fp);
		//}
		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::writeGraphParam2Bin(FILE *fp)
	{
		int node_size = nodes_.size();
		fwrite(&node_size, sizeof(int), 1, fp);

		for (int i = 0; i < nodes_.size(); i++)
		{
			// Use node name to verify
			const std::string node_name = nodes_[i]->getName();
			int name_len = node_name.length() + 1;
			fwrite(&name_len, sizeof(int), 1, fp);
			fwrite(node_name.c_str(), sizeof(char), name_len, fp);
			//fwrite(&index_, sizeof(int), 1, fp);

			const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[i]->getDataVec();

			int size = data_vec.size();
			fwrite(&size, sizeof(int), 1, fp);

			// data_vec[0] contains the cpu_data that should not be saved.
			if (size <= 1)
				continue ;

			//Dtype *testData = (Dtype *)malloc(sizeof(Dtype) * 12345);
			//memset(testData, 1, sizeof(Dtype) * 12345);
			for (int j = 1; j < size; j++)
			{
				int len = data_vec[j]->getSize()[tind::e4D];
				fwrite(&len, sizeof(int), 1, fp);
				//printf("w-len:%d\n", len);
				fwrite(data_vec[j]->getCpuData(), sizeof(Dtype), len, fp);	//

				//float *tdata = (float *)data_vec[j]->getCpuData();
				//for (int jj = 0; jj < len; jj++)
				//	printf("%f, ", *(tdata + jj));
			}
		}
		return 0;
	}
	template <typename Dtype>
	int Graph<Dtype>::readBin2GraphParam(FILE *fp)
	{
		// the variable node_size can shows how many nodes have been wrote there.
		int node_size = 0;
		fread(&node_size, sizeof(int), 1, fp);	

		for (int i = 0; i < node_size; i++)
		{
			int name_len = 0;
			fread(&name_len, sizeof(int), 1, fp);

			char *name = (char *)malloc(sizeof(char) * name_len);
			fread(name, sizeof(char), name_len, fp);
			printf("params name: %s\n", name);

			// Search all of the nodes in graph for finding the node that has the same name.
			for (int j = 0; j < nodes_.size(); j++)	// 加载结构时，需要检查重名的情况，确保到这里不会有重名node
			{
				// 思考id号如何使用，预训练，有相同名字且不同id时，如何处理？
				// 改为只用名字来连接输入输出，加载完毕后，重新为各个node生成新的id号。
				// id号即对应这nodes_的下标索引，用于索引相应的node
				if (!strcmp(name, nodes_[j]->getName().c_str()))
				{
					int size = 0;
					fread(&size, sizeof(int), 1, fp);

					if (size <= 1)
						break; 

					const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[j]->getDataVec();
					for (int k = 1; k < size; k++)
					{
						int len = 0;
						fread(&len, sizeof(int), 1, fp);
						fread(data_vec[k]->getCpuData(), sizeof(Dtype), len, fp);

						//float *tdata = (float *)data_vec[k]->getCpuData();
						//for (int jj = 0; jj < len; jj++)
						//	printf("%f, ", *(tdata + jj));
					}
					break;
				}
			}

			free(name);
		}
		return 0;
	}
	INSTANTIATE_CLASS(Graph);
}

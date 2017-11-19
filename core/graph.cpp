////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node graph, mainly contains nodes, 
//          support for node operation including forward and backward
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <sstream>
#include "graph.h"

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
	int Graph<Dtype>::getNodeIndex(const std::string &nodeName, int &index)
	{
		std::map<std::string, int>::iterator it = nodes_index_map_.find(nodeName);
		if (it == nodes_index_map_.end())
			return -1;

		index = it->second;
		return 0;
	}

	template <typename Dtype>
	void Graph<Dtype>::addNode(const std::string &nodeName,
		std::vector<std::shared_ptr<Op<Dtype>>> &op,
		std::vector<std::string> &inNodeNames = std::vector<std::string>())
	{
		//const auto layer_type = layer->getLayerType();
		//printf("NetWork addayer begin , type : %s\n", layer_type.c_str());
		std::shared_ptr<Node<Dtype>> node = std::make_shared<Node<Dtype>>();
		//node->inputs_index_.clear();
		//node->outputs_index_.clear();
		
		const int nodeIdx = nodes_.size();
		node->setIndex(nodeIdx);
		node->setName(nodeName);
		nodes_index_map_[nodeName] = nodeIdx;

		for (int i = 0; i < op.size(); i++)
			node->addSubOps(op[i]);

		node->setPhase(phase_);
		node->inferInteOp();
		//node->input_shape_ = data_[data_.size() - 1]->getShape();

		const std::shared_ptr<Op<Dtype>> inteOp = node->getInteOp();
		if (inteOp->getOpType() == "Input")
		{
			in_nodes_map_[nodeName] = nodeIdx;
		}
		else
		{
			for (int idx = 0; idx < inNodeNames.size(); idx++)
			{
				int index = -1;
				int ret = getNodeIndex(inNodeNames[idx], index);
				if (ret != 0)
				{
					DLOG_ERR("[ Graph::addNode ]: Can not get node with name %s \n", inNodeNames[idx].c_str());
					continue;
				}
				node->addInputName(inNodeNames[idx]); // Create link from current node to previous node
				node->addInputIdx(index);	
				nodes_[index]->addOutputName(node->getName()); // Create link from previous node to current node 
				nodes_[index]->addOutputIdx(nodeIdx);  
			}
			//node->input_shape_ = nodes_[node->inputs_index_[0]]->output_shape_;
			node->setInputShape(nodes_[node->getInputIdx()[0]]->getOutputShape());
			if (inteOp->getOpType() == "Output")
				out_nodes_map_[nodeName] = nodeIdx;
		}
		node->inferOutShape();
		node->initOp();
		node->initNode();
		nodes_.push_back(node);
		printf("NetWork addayer end. add data Tensor done.\n");
	}

	// fill the input data in innode->cpu_data_[0]
	template <typename Dtype>
	int Graph<Dtype>::setInNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> inputData, const std::vector<std::string> nodeNames)
	{
		if (inputData.size() != nodeNames.size())
		{ 
			DLOG_ERR("[ Graph::setInNode ]: inputData should have the same size with nodeNames\n");
			return -1;
		}
		if (inputData.size() <= 0)
		{
			DLOG_ERR("[ Graph::setInNode ]: inputData is invalid\n");
			return -1;
		}

		const int data_num = inputData[0]->getShape()[0];
		for (int i = 0; i < inputData.size(); i++)
		{
			if (data_num != inputData[i]->getShape()[0])
			{
				DLOG_ERR("[ Graph::setInNode ]: Each block of data should has the same num\n");
				return -1;
			}
		}

		for (int i = 0; i < inputData.size(); i++)
		{
			int index = -1;
			int ret = getNodeIndex(nodeNames[i], index);
			if (ret != 0)
			{
				DLOG_ERR("[ Graph::setInNode ]: Can not get node with name %s \n", nodeNames[i].c_str());
				return -1;
			}

			if (nodes_[index]->getDataVec()[0]->getSize()[tind::e4D] != inputData[i]->getSize()[tind::e4D])
			{
				nodes_[index]->setInputShape(inputData[i]->getShape());
				nodes_[index]->resetDataSize(0, inputData[i]->getShape());
				//nodes_[index]->inferOutShape();
				//cpu_data[0].reset(new Tensor<Dtype>(inputData[i]->getShape()));
			}
			inputData[i]->copyDataTo(*nodes_[index]->getDataVec()[0]);
		}
		return 0;
	}

	//fill the label data in outnode->cpu_data_[1]
	template <typename Dtype>
	int Graph<Dtype>::setOutNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> labelData, const std::vector<std::string> nodeNames)
	{
		if (labelData.size() != nodeNames.size())
		{
			DLOG_ERR("[ Graph::setOutNode ]: labelData should have the same size with nodeNames\n");
			return -1;
		}
		if (labelData.size() <= 0)
		{
			DLOG_ERR("[ Graph::setOutNode ]: labelData is invalid\n");
			return -1;
		}

		//printf("s0 set out node\n");
		for (int i = 0; i < labelData.size(); i++)
		{
			int index = -1;
			int ret = getNodeIndex(nodeNames[i], index);
			//printf("s1 set out node\n");
			if (ret != 0)
			{
				DLOG_ERR("[ Graph::setOutNode ]: Can not get node with name %s. \n", nodeNames[i].c_str());
				return -1;
			}
			
			// The format of output node is ( output[0], label[1], loss[2] )
			const int vecSize = nodes_[index]->getDataVec().size();
			if (vecSize != 3)
			{
				DLOG_ERR("[ Graph::setOutNode ]: Output node is not contains 3 tenosr. \n");
				return - 1;
			}
			/*if (nodes_[i]->inte_ops_->getOpDiff()[0]->getSize()[tind::e4D] != nodes_[i]->cpu_data_[0]->getSize()[tind::e4D])
				nodes_[i]->inte_ops_->getOpDiff()[0].reset(new Tensor<Dtype>(nodes_[i]->cpu_data_[0]->shape_));*/

			//printf("s set out node\n");
			if (nodes_[index]->getDataVec()[1]->getSize()[tind::e4D] != labelData[i]->getSize()[tind::e4D])
				nodes_[index]->resetDataSize(1, labelData[i]->getShape());

			const std::vector<std::shared_ptr<Tensor<Dtype>>> cpu_data = nodes_[index]->getDataVec();

			labelData[i]->copyDataTo(*cpu_data[1]);

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

			const std::shared_ptr<Op<Dtype>> inteOp_idx = nodes_[idx]->getInteOp();
			if (inteOp_idx->getOpType() == "Output")
				continue ;
			
			// recheck batch size
			const std::vector<int> idxOutShape = nodes_[idx]->getOutputShape();
			const std::vector<int> outputIdx  = nodes_[idx]->getOutputIdx();
			const std::vector<int> idxNextInShape = nodes_[outputIdx[0]]->getInputShape();
			if (idxOutShape != idxNextInShape)
			{
				nodes_[outputIdx[0]]->setInputShape(idxOutShape);
				//nodes_[outputIdx[0]]->inferOutShape();
				nodes_[outputIdx[0]]->resetDataSize(0, idxOutShape);	// it will call inferOutShape
				//nodes_[outputIdx[0]]->initOp();	// no, 前向和反向过程中，只有在用到时发现某块数据维度不对，才修改要用的那块内存大小。
			}

			// The bottom data of forward is saved in the node that executing forward operation. 
			nodes_[outputIdx[0]]->getDataVec()[0]->setZero();
			inteOp_idx->forward(nodes_[idx]->getDataVec(), nodes_[outputIdx[0]]->getDataVec());

			//float *outdata0 = (float *)nodes_[idx]->getDataVec()[1]->getData();
			//for (int j = 0; j < nodes_[idx]->getDataVec()[1]->getSize()[3]; j++)
			//	printf("%f,", outdata0[j]);
			//printf("\n");

			for (int i = 0; i < outputIdx.size(); i++)
			{
				//printf("push %d\n", nodes_[idx]->outputs_index_[i]);
				nodes_idx_stack_.push(outputIdx[i]);
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

			const std::vector<int> inputIdx = nodes_[idx]->getInputIdx();
			for (int i = 0; i < inputIdx.size(); i++)
			{
				//printf("push %d\n", nodes_[idx]->inputs_index_[i]);
				nodes_idx_stack_.push(inputIdx[i]);
			}

			const std::shared_ptr<Op<Dtype>> inteOp_idx = nodes_[idx]->getInteOp();
			const std::vector<std::shared_ptr<Tensor<Dtype>>> data_idx = nodes_[idx]->getDataVec();
		
			// recheck batch size : diff_
			if (inteOp_idx->getOpDiff()[0]->getSize()[tind::e4D] != data_idx[0]->getSize()[tind::e4D])
				inteOp_idx->getOpDiff()[0].reset(new Tensor<Dtype>(data_idx[0]->getShape()));

			if (inteOp_idx->getOpType() == "Output")
				continue;

			// The bottom data of backward is saved in the node that executing backward operation. 
			const std::vector<int> outputIdx = nodes_[idx]->getOutputIdx();
			inteOp_idx->getOpDiff()[0]->setZero();
			inteOp_idx->backward(data_idx, nodes_[outputIdx[0]]->getDataVec(),
				inteOp_idx->getOpDiff(), nodes_[outputIdx[0]]->getInteOp()->getOpDiff());
		}

		//// 注意 当前为取[0]号输出，后面支持多输出（兼容前向？或在计算图模型中处理？）！！！

		//for (int i = (int)(nodes_.size()) - 2; i >= 0; i--)
		//{
		//	//printf("backward[%d].\n", i);
		//	timer.Start();
		//	nodes_[i]->inte_ops_->getOpDiff()[0]->setZero();
		//	nodes_[i]->inte_ops_->backward(nodes_[i]->cpu_data_, nodes_[nodes_[i]->outputs_index_[0]]->cpu_data_,
		//		nodes_[i]->inte_ops_->getOpDiff(), nodes_[nodes_[i]->outputs_index_[0]]->inte_ops_->getOpDiff());
		//	float us = timer.MilliSeconds();
		//	printf("time[%d]: %f ms\n", i, us);
		//}
		return 0;//nodes_[nodes_.size() - 1]->cpu_data_[2];
	}

	template <typename Dtype>
	int Graph<Dtype>::getLoss(const std::string &nodeName, Dtype &loss)
	{
		loss = 100.0;

		int index = -1;
		int ret = getNodeIndex(nodeName, index);
		if (ret != 0)
		{
			DLOG_ERR("[ Graph::getLoss ]: Can not get node with name < %s >.\n", nodeName.c_str());
			return -1;
		}
		if (nodes_[index]->getInteOp()->getOpType() != "Output")
		{
			DLOG_ERR("[ Graph::getLoss ]: The node with name < %s >, is not an output node.\n", nodeName.c_str());
			return -1;
		}
		loss = *(Dtype *)(nodes_[index]->getDataVec()[2]->getData());

		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::getNodeData(const std::string &nodeName, std::shared_ptr<Tensor<Dtype>> &cpuData)
	{
		int index = -1;
		int ret = getNodeIndex(nodeName, index);
		if (ret != 0)
		{
			DLOG_ERR("[ Graph::getNodeData ]: Can not get node with name < %s >.\n", nodeName.c_str());
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
			
			const std::string curNodeName = nodes_[idx]->getName();
			const std::shared_ptr<Op<Dtype>> inteOp_idx = nodes_[idx]->getInteOp();
			printf("*  node name: <%s> , op type: <%s>.\n", curNodeName.c_str(), inteOp_idx->getOpType().c_str());

			// weight / blas
			const std::vector<std::shared_ptr<Tensor<Dtype>>> dataVec = nodes_[idx]->getDataVec();
			const std::vector<int> dataShape = dataVec[0]->getShape();
			printf("*  data: (%d, %d, %d, %d). \n", dataShape[tind::eNum], dataShape[tind::eChannels], dataShape[tind::eHeight], dataShape[tind::eWidth]);
			if (dataVec.size() >= 2)
			{
				const std::vector<int> weightShape = dataVec[1]->getShape();
				printf("*  weight: (%d, %d, %d, %d). \n", weightShape[tind::eNum], weightShape[tind::eChannels], weightShape[tind::eHeight], weightShape[tind::eWidth]);
				
				if (dataVec.size() >= 3)
				{
					const std::vector<int> blasShape = dataVec[2]->getShape();
					printf("*  blas: (%d, %d, %d, %d). \n", blasShape[tind::eNum], blasShape[tind::eChannels], blasShape[tind::eHeight], blasShape[tind::eWidth]);
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
			const std::vector<std::shared_ptr<Tensor<Dtype>>> gradientVec = inteOp_idx->getOpGradient();
			if (gradientVec.size() != 0)
			{
				const std::vector<int> shape = gradientVec[0]->getShape();
				printf("*  gradient: (%d, %d, %d, %d). \n", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}
			else
				printf("*  gradient: None. \n");

			const std::vector<std::shared_ptr<Tensor<Dtype>>> diffVec = inteOp_idx->getOpDiff();
			if (diffVec.size() != 0)
			{
				const std::vector<int> shape = diffVec[0]->getShape();
				printf("*  diff: (%d, %d, %d, %d). \n", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}
			else
				printf("*  diff: None. \n");

			// input / output
			const std::vector<int> inputIdx = nodes_[idx]->getInputIdx();
			const std::vector<int> outputIdx = nodes_[idx]->getOutputIdx();
			for (int i = 0; i < inputIdx.size(); i++)
			{
				const std::vector<int> shape = nodes_[inputIdx[i]]->getOutputShape(); // getDataVec()[0]->getShape();
				printf("*  %s <%s> (%d, %d, %d, %d) -> %s. \n", nodes_[inputIdx[i]]->getName().c_str(),
					nodes_[inputIdx[i]]->getInteOp()->getOpType().c_str(),
					shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth],
					curNodeName.c_str());
			}
			for (int i = 0; i < outputIdx.size(); i++)
			{
				const std::vector<int> shape = nodes_[outputIdx[i]]->getInputShape(); // getDataVec()[0]->getShape();
				printf("*  %s -> %s <%s> (%d, %d, %d, %d). \n", curNodeName.c_str(),
					nodes_[outputIdx[i]]->getName().c_str(),
					nodes_[outputIdx[i]]->getInteOp()->getOpType().c_str(),
					shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
			}

			for (int i = 0; i < outputIdx.size(); i++)
				nodes_idx_stack_.push(outputIdx[i]);
		}
		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::writeGraph2Text(FILE *fp)
	{
		std::stringstream optss;
		optss << "graphSize:" << nodes_.size() << ";";
		fprintf(fp, "%s\n", optss.str().c_str());

		for (int i = 0; i < nodes_.size(); i++)
		{
			//nodes_[i]->writeNode2Text(fp);
			////name, in_idx（save name）
			std::stringstream ss;
			ss << "nodeName:" << nodes_[i]->getName() << ",opType:" << nodes_[i]->getInteOp()->getOpType() << "," << "inNodesName:";

			const std::vector<std::string> input_names = nodes_[i]->getInputName();
			ss << "(";
			for (int j = 0; j < input_names.size(); j++)
			{
				ss << input_names[j];
				if (j < input_names.size() - 1)
					ss << ",";
			}
			ss << ");" << std::endl;

			std::string opParam = nodes_[i]->getOpParamBufStr();
			ss << opParam;

			fprintf(fp, "%s\n", ss.str().c_str());
		}
		return 0;
	}

	template <typename Dtype>
	int Graph<Dtype>::readText2Graph(FILE *fp)
	{
		nodes_.clear();

		char cc[1000];

		int graphSize = 0;
		if (EOF != fscanf(fp, "%s", cc))	// Fetch the first line to get the graph size.
		{
			std::string cstr(cc);
			printf("read0: %s\n", cstr.c_str());
			graphSize = atoi(fetchSubStr(cstr, "graphSize:", ";").c_str());
			printf("read0: %d\n", graphSize);
		}
		std::string nodeName, opType;
		std::vector<std::string> inNodeNames;
		int lineCount = 0;
		while (lineCount < graphSize * 2)
		{
			if (EOF == fscanf(fp, "%s", cc))
				return -1;
			lineCount++;

			// Fetch each node's parameters
			std::string cstr(cc);
			if (lineCount % 2 == 1)          // Current and input nodes' name 
			{
				printf("read1: %s\n", cstr.c_str());

				// Fetch node name
				nodeName = fetchSubStr(cstr, "nodeName:", ",");

				// Fetch operator's name in this node
				opType = fetchSubStr(cstr, "opType:", ",");
				printf("read11_0: %s, %s\n", nodeName.c_str(), opType.c_str());

				if (opType == "Input")
					continue;

				// Fetch input nodes' name of this node
				std::string inNamesStr = fetchSubStr(cstr, "inNodesName:(", ")");
				int commaFlag;
				inNodeNames.clear();
				while ((commaFlag = inNamesStr.find(",")) != -1)
				{
					std::string name = inNamesStr.substr(0, commaFlag);
					inNodeNames.push_back(name);
					inNamesStr = inNamesStr.substr(commaFlag + 1, inNamesStr.length());
				}
				inNodeNames.push_back(inNamesStr);
				for (int i = 0; i < inNodeNames.size(); i++)
					printf("inNodeNames[%d]: %s\n", i, inNodeNames[i].c_str());

			}
			else
			{
				// OpParams have been saved in this line
				printf("read2: %s\n", cstr.c_str());

				// Create node here according to previous information.
				std::shared_ptr<dlex_cnn::Op<Dtype>> nodeOp = dlex_cnn::OpFactory<Dtype>::getInstance().createOpByType(opType);
				if (nodeOp == NULL)
					DLOG_ERR("[ Graph<Dtype>::readText2Graph ]: Can not create Op by type (%s) \n", opType.c_str());
				nodeOp->setOpParam(cstr);
				std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > nodeOps;
				nodeOps.push_back(nodeOp);

				this->addNode(nodeName, nodeOps, inNodeNames);
			}

		}
		//std::stringstream optss;
		//optss << "graphSize:" << nodes_.size() << ";";
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
		int nodeSize = nodes_.size();
		fwrite(&nodeSize, sizeof(int), 1, fp);

		for (int i = 0; i < nodes_.size(); i++)
		{
			// Use node name to verify
			const std::string nodeName = nodes_[i]->getName();
			int nameLen = nodeName.length() + 1;
			fwrite(&nameLen, sizeof(int), 1, fp);
			fwrite(nodeName.c_str(), sizeof(char), nameLen, fp);
			//fwrite(&index_, sizeof(int), 1, fp);

			const std::vector<std::shared_ptr<Tensor<Dtype>>> dataVec = nodes_[i]->getDataVec();

			int size = dataVec.size();
			fwrite(&size, sizeof(int), 1, fp);

			// dataVec[0] contains the cpu_data that should not be saved.
			if (size <= 1)
				continue ;

			//Dtype *testData = (Dtype *)malloc(sizeof(Dtype) * 12345);
			//memset(testData, 1, sizeof(Dtype) * 12345);
			for (int j = 1; j < size; j++)
			{
				int len = dataVec[j]->getSize()[tind::e4D];
				fwrite(&len, sizeof(int), 1, fp);
				//printf("w-len:%d\n", len);
				fwrite(dataVec[j]->getData(), sizeof(Dtype), len, fp);	//

				//float *tdata = (float *)dataVec[j]->getData();
				//for (int jj = 0; jj < len; jj++)
				//	printf("%f, ", *(tdata + jj));
			}
		}
		return 0;
	}
	template <typename Dtype>
	int Graph<Dtype>::readBin2GraphParam(FILE *fp)
	{
		// the variable nodeSize can shows how many nodes have been wrote there.
		int nodeSize = 0;
		fread(&nodeSize, sizeof(int), 1, fp);	

		for (int i = 0; i < nodeSize; i++)
		{
			int nameLen = 0;
			fread(&nameLen, sizeof(int), 1, fp);

			char *name = (char *)malloc(sizeof(char) * nameLen);
			fread(name, sizeof(char), nameLen, fp);
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

					const std::vector<std::shared_ptr<Tensor<Dtype>>> dataVec = nodes_[j]->getDataVec();
					for (int k = 1; k < size; k++)
					{
						int len = 0;
						fread(&len, sizeof(int), 1, fp);
						fread(dataVec[k]->getData(), sizeof(Dtype), len, fp);

						//float *tdata = (float *)dataVec[k]->getData();
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
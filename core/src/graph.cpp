////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node graph, mainly contains nodes, 
//          support for node operation including Forward and Backward
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <sstream>
#include "graph.h"
#include "task.h"
#include "util/math_functions.h"

namespace dlex_cnn {
template <typename Dtype>
Graph<Dtype>::Graph() {}

template <typename Dtype>
Graph<Dtype>::~Graph() {}

template <typename Dtype>
int Graph<Dtype>::GetNodeIndex(const std::string &node_name, int &index) {
  std::map<std::string, int>::iterator it = nodes_index_map_.find(node_name);
  if (it == nodes_index_map_.end())
    return -1;

  index = it->second;
  return 0;
}

template <typename Dtype>
void Graph<Dtype>::AddNode(const std::string &node_name,
  const std::vector< std::shared_ptr< Op<Dtype> > > &op,
  const std::vector<std::string> &in_node_names) {
  std::shared_ptr<Node<Dtype>> node = std::make_shared<Node<Dtype>>();

  const int node_idx = nodes_.size();
  node->set_index(node_idx);
  node->set_name(node_name);
  nodes_index_map_[node_name] = node_idx;

  for (int i = 0; i < op.size(); i++)
    node->add_sub_ops(op[i]);

  node->set_phase(phase_);
  node->InferInteOp();
  //node->input_shape_ = cpu_data_[cpu_data_.size() - 1]->get_shape();

  const std::shared_ptr<Op<Dtype>> inteOp = node->get_inte_op();
  if (inteOp->get_op_type() == "Input") {
    in_nodes_map_[node_name] = node_idx;
  }
  else {
    for (int idx = 0; idx < in_node_names.size(); idx++) {
      int index = -1;
      int ret = GetNodeIndex(in_node_names[idx], index);
      if (ret != 0) {
        DLOG_ERR("[ Graph::AddNode ]: Can not get node with name %s.", in_node_names[idx].c_str());
        continue;
      }
      node->add_input_name(in_node_names[idx]); // Create link from current node to previous node
      node->add_input_idx(index);
      nodes_[index]->add_output_name(node->get_name()); // Create link from previous node to current node 
      nodes_[index]->add_output_idx(node_idx);
    }
    //node->input_shape_ = nodes_[node->inputs_index_[0]]->output_shape_;
    node->set_input_shape(nodes_[node->get_input_idx()[0]]->get_output_shape());
    if (inteOp->get_op_type() == "Output")
      out_nodes_map_[node_name] = node_idx;
  }
  node->InferOutShape();
  node->InitOp();
  node->InitNode();
  nodes_.push_back(node);
  DLOG_INFO("Add node: %s.", node->get_inte_op()->get_op_type().c_str());
}

// Initialize all of the weight and bias.
template <typename Dtype>
void Graph<Dtype>::ParamsInit() {
  for (int i = 0; i < nodes_.size(); i++) {
    const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[i]->get_data_vec();

    if (Task::mode() == tind::CPU) {
      if (data_vec.size() > 1) {
        normal_distribution_init<Dtype>(data_vec[1]->get_size()[tind::e4D], 0.0f, 0.1f, (Dtype *)data_vec[1]->GetPushCpuData());
        if (data_vec.size() > 2)
          set_cpu<Dtype>(data_vec[2]->get_size()[tind::e4D], 0.0f, (Dtype *)data_vec[2]->GetPushCpuData());
      }
    }
    else {
#ifdef USE_CUDA
      if (data_vec.size() > 1) {
        dlex_gpu_rng_gaussian<Dtype>(data_vec[1]->get_size()[tind::e4D], 0.0f, 0.1f, (Dtype *)data_vec[1]->GetPushGpuData());
        if (data_vec.size() > 2)
          set_gpu<Dtype>(data_vec[2]->get_size()[tind::e4D], 0.0f, (Dtype *)data_vec[2]->GetPushGpuData());
      }
#else
      DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif
    }
  }
}

template <typename Dtype>
int Graph<Dtype>::SetIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names) {
  in_node_names_.assign(in_node_names.begin(), in_node_names.end());
  out_node_names_.assign(out_node_names.begin(), out_node_names.end());

  return 0;
}

// fill the input data in innode->cpu_data_[0]
template <typename Dtype>
int Graph<Dtype>::SetInNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> input_data) {
  if (input_data.size() != in_node_names_.size()) {
    DLOG_ERR("[ Graph::SetInNode ]: input_data should have the same size with node_names.");
    return -1;
  }
  if (input_data.size() <= 0) {
    DLOG_ERR("[ Graph::SetInNode ]: input_data is invalid.");
    return -1;
  }

  const int data_num = input_data[0]->get_shape()[0];
  for (int i = 0; i < input_data.size(); i++) {
    if (data_num != input_data[i]->get_shape()[0]) {
      DLOG_ERR("[ Graph::SetInNode ]: Each block of data should has the same num.");
      return -1;
    }
  }

  for (int i = 0; i < input_data.size(); i++) {
    int index = -1;
    int ret = GetNodeIndex(in_node_names_[i], index);
    if (ret != 0) {
      DLOG_ERR("[ Graph::SetInNode ]: Can not get node with name %s.", in_node_names_[i].c_str());
      return -1;
    }

    if (nodes_[index]->get_data_vec()[0]->get_size()[tind::e4D] != input_data[i]->get_size()[tind::e4D]) {
      nodes_[index]->set_input_shape(input_data[i]->get_shape());
      nodes_[index]->ResetDataSize(0, input_data[i]->get_shape());
      //nodes_[index]->InferOutShape();
      //cpu_data[0].reset(new Tensor<Dtype>(input_data[i]->get_shape()));
    }
    if (Task::mode() == tind::CPU)
      input_data[i]->CopyDataTo(*nodes_[index]->get_data_vec()[0], tind::eHost2Host);
    else
      input_data[i]->CopyDataTo(*nodes_[index]->get_data_vec()[0], tind::eDevice2Device);
  }
  return 0;
}

//fill the label data in outnode->cpu_data_[1]
template <typename Dtype>
int Graph<Dtype>::SetOutNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> label_data) {
  if (label_data.size() != out_node_names_.size()) {
    DLOG_ERR("[ Graph::SetOutNode ]: label_data should have the same size with node_names.");
    return -1;
  }
  if (label_data.size() <= 0) {
    DLOG_ERR("[ Graph::SetOutNode ]: label_data is invalid.");
    return -1;
  }

  //printf("s0 set out node\n");
  for (int i = 0; i < label_data.size(); i++) {
    int index = -1;
    int ret = GetNodeIndex(out_node_names_[i], index);
    //printf("s1 set out node\n");
    if (ret != 0) {
      DLOG_ERR("[ Graph::SetOutNode ]: Can not get node with name %s.", out_node_names_[i].c_str());
      return -1;
    }

    // The format of output node is ( output[0], label[1], loss[2] )
    const int vec_size = nodes_[index]->get_data_vec().size();
    if (vec_size != 3) {
      DLOG_ERR("[ Graph::SetOutNode ]: Output node is not contains 3 tenosr.");
      return -1;
    }
    /*if (nodes_[i]->inte_ops_->get_op_diff()[0]->get_size()[tind::e4D] != nodes_[i]->cpu_data_[0]->get_size()[tind::e4D])
      nodes_[i]->inte_ops_->get_op_diff()[0].reset(new Tensor<Dtype>(nodes_[i]->cpu_data_[0]->shape_));*/

    //printf("s set out node\n");
    if (nodes_[index]->get_data_vec()[1]->get_size()[tind::e4D] != label_data[i]->get_size()[tind::e4D])
      nodes_[index]->ResetDataSize(1, label_data[i]->get_shape());

    if (Task::mode() == tind::CPU)
      label_data[i]->CopyDataTo(*nodes_[index]->get_data_vec()[1], tind::eHost2Host);
    else
      label_data[i]->CopyDataTo(*nodes_[index]->get_data_vec()[1], tind::eDevice2Device);
  }
  return 0;
}

// Forward graph by DFS
template <typename Dtype>
int Graph<Dtype>::ForwardGraph() {
  //printf("NetWork Forward begin.\n");
  if (nodes_.size() <= 1) {
    DLOG_ERR("[ Graph::ForwardGraph ]: A graph should has more than 2 nodes.");
    return -1;
  }
  if (in_nodes_map_.size() <= 0 || out_nodes_map_.size() <= 0) {
    DLOG_ERR("[ Graph::ForwardGraph ]: input node or output node is empty.");
    return -1;
  }

  while (!nodes_idx_stack_.empty())
    nodes_idx_stack_.pop();

  std::map<std::string, int>::iterator it = in_nodes_map_.begin();	
  // push all of the input nodes
  while (it != in_nodes_map_.end()) {
    //printf("while (it != in_nodes_map_.end())\n");
    //it->first; it->second;
    nodes_idx_stack_.push(it->second);
    it++;
  }

  //// 注意 当前为取[0]号输出，后面支持多输出（兼容前向？或在计算图模型中处理？）
  //DFS , 后需添加depends
  while (!nodes_idx_stack_.empty()) {
    //printf("!nodes_idx_stack_.empty()\n");
    int idx = nodes_idx_stack_.top();
    nodes_idx_stack_.pop();

    const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->get_inte_op();
    if (inte_op_idx->get_op_type() == "Output")
      continue;

    // recheck batch size
    const std::vector<int> idx_out_shape = nodes_[idx]->get_output_shape();
    const std::vector<int> output_idx = nodes_[idx]->get_output_idx();
    const std::vector<int> idx_next_in_shape = nodes_[output_idx[0]]->get_input_shape();
    if (idx_out_shape != idx_next_in_shape) {
      nodes_[output_idx[0]]->set_input_shape(idx_out_shape);
      //nodes_[output_idx[0]]->InferOutShape();
      nodes_[output_idx[0]]->ResetDataSize(0, idx_out_shape);	// it will call InferOutShape
      //nodes_[output_idx[0]]->InitOp();	// no, 前向和反向过程中，只有在用到时发现某块数据维度不对，才修改要用的那块内存大小。
    }

    // The bottom data of Forward is saved in the node that executing Forward operation. 
    if (Task::mode() == tind::CPU) {
      //nodes_[output_idx[0]]->get_data_vec()[0]->SetCpuZero();
      inte_op_idx->Forward(nodes_[idx]->get_data_vec(), nodes_[output_idx[0]]->get_data_vec());
    }
    else {
#ifdef USE_CUDA
      nodes_[output_idx[0]]->get_data_vec()[0]->SetGpuZero();
      inte_op_idx->Forward_gpu(nodes_[idx]->get_data_vec(), nodes_[output_idx[0]]->get_data_vec());
#else
      DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif
    }

    //float *outdata0 = (float *)nodes_[idx]->get_data_vec()[1]->GetPushCpuData();
    //for (int j = 0; j < nodes_[idx]->get_data_vec()[1]->get_size()[3]; j++)
    //	printf("%f,", outdata0[j]);
    //printf("\n");

    for (int i = 0; i < output_idx.size(); i++) {
      //printf("push %d\n", nodes_[idx]->outputs_index_[i]);
      nodes_idx_stack_.push(output_idx[i]);
    }
  }

  return 0;
}

// Backward graph by DFS
template <typename Dtype>
int Graph<Dtype>::BackwardGraph() {
  if (nodes_.size() <= 1) {
    DLOG_ERR("[ Graph::BackwardGraph ]: A graph should has more than 2 nodes.");
    return -1;
  }
  if (in_nodes_map_.size() <= 0 || out_nodes_map_.size() <= 0) {
    DLOG_ERR("[ Graph::BackwardGraph ]: input node or output node is empty.");
    return -1;
  }

  Timer timer;
  while (!nodes_idx_stack_.empty())
    nodes_idx_stack_.pop();

  std::map<std::string, int>::iterator it = out_nodes_map_.begin();	
  // push all of the input node
  while (it != out_nodes_map_.end()) {
    //printf("while (it != out_nodes_map_.end())\n");
    //it->first; it->second;
    nodes_idx_stack_.push(it->second);
    it++;
  }

	//DFS , 后需添加depends
  while (!nodes_idx_stack_.empty()) {
    int idx = nodes_idx_stack_.top();
    //printf("Backward idx = %d, input_idx = %d\n", idx, nodes_[idx]->inputs_index_.size());
    nodes_idx_stack_.pop();

    const std::vector<int> input_idx = nodes_[idx]->get_input_idx();
    for (int i = 0; i < input_idx.size(); i++) {
      //printf("push %d\n", nodes_[idx]->inputs_index_[i]);
      nodes_idx_stack_.push(input_idx[i]);
    }

    const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->get_inte_op();
    const std::vector<std::shared_ptr<Tensor<Dtype>>> data_idx = nodes_[idx]->get_data_vec();

    // recheck batch size : diff_
    if (inte_op_idx->get_op_diff()[0]->get_size()[tind::e4D] != data_idx[0]->get_size()[tind::e4D])
      inte_op_idx->get_op_diff()[0].reset(new Tensor<Dtype>(data_idx[0]->get_shape()));

    if (inte_op_idx->get_op_type() == "Output")
      continue;

    // The bottom data of Backward is saved in the node that executing Backward operation. 
    const std::vector<int> output_idx = nodes_[idx]->get_output_idx();
    if (Task::mode() == tind::CPU) {
      //inte_op_idx->get_op_diff()[0]->SetCpuZero();
      inte_op_idx->Backward(data_idx, nodes_[output_idx[0]]->get_data_vec(),
        inte_op_idx->get_op_diff(), nodes_[output_idx[0]]->get_inte_op()->get_op_diff());
    }
    else {
#ifdef USE_CUDA
      inte_op_idx->get_op_diff()[0]->SetGpuZero();
      inte_op_idx->Backward_gpu(data_idx, nodes_[output_idx[0]]->get_data_vec(),
        inte_op_idx->get_op_diff(), nodes_[output_idx[0]]->get_inte_op()->get_op_diff());
#else
      DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif	
    }

  }
  //// 注意 当前为取[0]号输出，后面支持多输出?
  return 0;
}

template <typename Dtype>
int Graph<Dtype>::GetLoss(const std::string &node_name, Dtype &loss) {
  loss = 100.0;

  int index = -1;
  int ret = GetNodeIndex(node_name, index);
  if (ret != 0) {
    DLOG_ERR("[ Graph::GetLoss ]: Can not get node with name < %s >.", node_name.c_str());
    return -1;
  }
  if (nodes_[index]->get_inte_op()->get_op_type() != "Output") {
    DLOG_ERR("[ Graph::GetLoss ]: The node with name < %s >, is not an output node.", node_name.c_str());
    return -1;
  }
  loss = *(Dtype *)(nodes_[index]->get_data_vec()[2]->GetPushCpuData());

  return 0;
}

template <typename Dtype>
int Graph<Dtype>::GetNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &data) {
  int index = -1;
  int ret = GetNodeIndex(node_name, index);
  if (ret != 0) {
    DLOG_ERR("[ Graph::GetNodeData ]: Can not get node with name < %s >.", node_name.c_str());
    return -1;
  }
  data = nodes_[index]->get_data_vec()[0];

  return 0;
}

template <typename Dtype>
int Graph<Dtype>::GraphShow() {
  //printf("NetWork Forward begin.\n");
  if (nodes_.size() <= 1) {
    DLOG_ERR("[ Graph::ForwardGraph ]: A graph should has more than 2 nodes.");
    return -1;
  }

  while (!nodes_idx_stack_.empty())
    nodes_idx_stack_.pop();

  std::map<std::string, int>::iterator it = in_nodes_map_.begin();
  // push all of the input node
  while (it != in_nodes_map_.end()) {
    //it->first; it->second;
    nodes_idx_stack_.push(it->second);
    it++;
  }

  //DFS , 后需添加depends
  while (!nodes_idx_stack_.empty()) {
    int idx = nodes_idx_stack_.top();
    DLOG_INFO("================================================(%d)==", idx);
    nodes_idx_stack_.pop();

    const std::string cur_node_name = nodes_[idx]->get_name();
    const std::shared_ptr<Op<Dtype>> inte_op_idx = nodes_[idx]->get_inte_op();
    DLOG_INFO("*  node name: <%s> , op type: <%s>.", cur_node_name.c_str(), inte_op_idx->get_op_type().c_str());

    // weight / blas
    const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[idx]->get_data_vec();
    const std::vector<int> data_shape = data_vec[0]->get_shape();
    DLOG_INFO("*  data: (%d, %d, %d, %d).", data_shape[tind::eNum], data_shape[tind::eChannels], data_shape[tind::eHeight], data_shape[tind::eWidth]);
    if (data_vec.size() >= 2) {
      const std::vector<int> weight_shape = data_vec[1]->get_shape();
      DLOG_INFO("*  weight: (%d, %d, %d, %d).", weight_shape[tind::eNum], weight_shape[tind::eChannels], weight_shape[tind::eHeight], weight_shape[tind::eWidth]);

      if (data_vec.size() >= 3) {
        const std::vector<int> blas_shape = data_vec[2]->get_shape();
        DLOG_INFO("*  blas: (%d, %d, %d, %d).", blas_shape[tind::eNum], blas_shape[tind::eChannels], blas_shape[tind::eHeight], blas_shape[tind::eWidth]);
      }
      else
        DLOG_INFO("*  blas: None. ");
    }
    else {
      DLOG_INFO("*  weight: None.");
      DLOG_INFO("*  blas: None.");
    }

    // gradient / diff
    const std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_vec = inte_op_idx->get_op_gradient();
    if (gradient_vec.size() != 0) {
      const std::vector<int> shape = gradient_vec[0]->get_shape();
      DLOG_INFO("*  gradient: (%d, %d, %d, %d).", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
    }
    else
      DLOG_INFO("*  gradient: None.");

    const std::vector<std::shared_ptr<Tensor<Dtype>>> diff_vec = inte_op_idx->get_op_diff();
    if (diff_vec.size() != 0) {
      const std::vector<int> shape = diff_vec[0]->get_shape();
      DLOG_INFO("*  diff: (%d, %d, %d, %d).", shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
    }
    else
      DLOG_INFO("*  diff: None.");

    // input / output
    const std::vector<int> input_idx = nodes_[idx]->get_input_idx();
    const std::vector<int> output_idx = nodes_[idx]->get_output_idx();
    for (int i = 0; i < input_idx.size(); i++) {
      const std::vector<int> shape = nodes_[input_idx[i]]->get_output_shape(); // get_data_vec()[0]->get_shape();
      DLOG_INFO("*  %s <%s> (%d, %d, %d, %d) -> %s.", nodes_[input_idx[i]]->get_name().c_str(),
        nodes_[input_idx[i]]->get_inte_op()->get_op_type().c_str(),
        shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth],
        cur_node_name.c_str());
    }
    for (int i = 0; i < output_idx.size(); i++) {
      const std::vector<int> shape = nodes_[output_idx[i]]->get_input_shape(); // get_data_vec()[0]->get_shape();
      DLOG_INFO("*  %s -> %s <%s> (%d, %d, %d, %d).", cur_node_name.c_str(),
        nodes_[output_idx[i]]->get_name().c_str(),
        nodes_[output_idx[i]]->get_inte_op()->get_op_type().c_str(),
        shape[tind::eNum], shape[tind::eChannels], shape[tind::eHeight], shape[tind::eWidth]);
    }

    for (int i = 0; i < output_idx.size(); i++)
      nodes_idx_stack_.push(output_idx[i]);
  }
  return 0;
}

template <typename Dtype>
int Graph<Dtype>::WriteGraph2Text(FILE *fp) {
  std::stringstream optss;
  optss << "nodes_size:" << nodes_.size() << ";";
  fprintf(fp, "%s\n", optss.str().c_str());

  for (int i = 0; i < nodes_.size(); i++) {
    //nodes_[i]->writeNode2Text(fp);
    ////name, in_idx（save name）
    std::stringstream ss;
    ss << "node_name:" << nodes_[i]->get_name() << ",op_type:" << nodes_[i]->get_inte_op()->get_op_type() << "," << "inNodesName:";

    const std::vector<std::string> input_names = nodes_[i]->get_input_name();
    ss << "(";
    for (int j = 0; j < input_names.size(); j++) {
      ss << input_names[j];
      if (j < input_names.size() - 1)
        ss << ",";
    }
    ss << ");" << std::endl;

    std::string op_param = nodes_[i]->GetOpParamBufStr();
    ss << op_param;

    fprintf(fp, "%s\n", ss.str().c_str());
  }
  return 0;
}

template <typename Dtype>
int Graph<Dtype>::ReadText2Graph(FILE *fp) {
  nodes_.clear();

  char cc[1000];

  int graph_size = 0;
  // Fetch the first line to get the graph size.
  if (EOF != fscanf(fp, "%s", cc)) {
    std::string cstr(cc);
    printf("read0: %s\n", cstr.c_str());
    graph_size = atoi(FetchSubStr(cstr, "nodes_size:", ";").c_str());
    printf("read0: %d\n", graph_size);
  }
  std::string node_name, op_type;
  std::vector<std::string> in_node_names;
  int line_count = 0;
  while (line_count < graph_size * 2) {
    if (EOF == fscanf(fp, "%s", cc))
      return -1;
    line_count++;

    // Fetch each node's parameters
    std::string cstr(cc);  
    // Current and input nodes' name 
    if (line_count % 2 == 1) {
      printf("read1: %s\n", cstr.c_str());

      // Fetch node name
      node_name = FetchSubStr(cstr, "node_name:", ",");

      // Fetch operator's name in this node
      op_type = FetchSubStr(cstr, "op_type:", ",");
      printf("read11_0: %s, %s\n", node_name.c_str(), op_type.c_str());

      if (op_type == "Input")
        continue;

      // Fetch input nodes' name of this node
      std::string inNamesStr = FetchSubStr(cstr, "inNodesName:(", ")");
      int comma_flag;
      in_node_names.clear();
      while ((comma_flag = inNamesStr.find(",")) != -1) {
        std::string name = inNamesStr.substr(0, comma_flag);
        in_node_names.push_back(name);
        inNamesStr = inNamesStr.substr(comma_flag + 1, inNamesStr.length());
      }
      in_node_names.push_back(inNamesStr);
      for (int i = 0; i < in_node_names.size(); i++)
        printf("in_node_names[%d]: %s\n", i, in_node_names[i].c_str());

    }
    else {
      // OpParams have been saved in this line
      printf("read2: %s\n", cstr.c_str());

      // Create node here according to previous information.
      std::shared_ptr<dlex_cnn::Op<Dtype>> node_op = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType(op_type);
      if (node_op == NULL)
        DLOG_ERR("[ Graph<Dtype>::ReadText2Graph ]: Can not create Op by type (%s).", op_type.c_str());
      node_op->SetOpParam(cstr);
      std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > node_ops;
      node_ops.push_back(node_op);

      this->AddNode(node_name, node_ops, in_node_names);
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
int Graph<Dtype>::WriteGraphParam2Bin(FILE *fp) {
  int node_size = nodes_.size();
  fwrite(&node_size, sizeof(int), 1, fp);

  for (int i = 0; i < nodes_.size(); i++) {
    // Use node name to verify
    const std::string node_name = nodes_[i]->get_name();
    int name_len = node_name.length() + 1;
    fwrite(&name_len, sizeof(int), 1, fp);
    fwrite(node_name.c_str(), sizeof(char), name_len, fp);
    //fwrite(&index_, sizeof(int), 1, fp);

    const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[i]->get_data_vec();

    int size = data_vec.size();
    fwrite(&size, sizeof(int), 1, fp);

    // data_vec[0] contains the cpu_data that should not be saved.
    if (size <= 1)
      continue;

    //Dtype *testData = (Dtype *)malloc(sizeof(Dtype) * 12345);
    //memset(testData, 1, sizeof(Dtype) * 12345);
    for (int j = 1; j < size; j++) {
      int len = data_vec[j]->get_size()[tind::e4D];
      fwrite(&len, sizeof(int), 1, fp);
      //printf("w-len:%d\n", len);
      fwrite(data_vec[j]->GetPushCpuData(), sizeof(Dtype), len, fp);	//

      //float *tdata = (float *)data_vec[j]->GetPushCpuData();
      //for (int jj = 0; jj < len; jj++)
      //	printf("%f, ", *(tdata + jj));
    }
  }
  return 0;
}

template <typename Dtype>
int Graph<Dtype>::ReadBin2GraphParam(FILE *fp) {
  // the variable node_size can shows how many nodes have been wrote there.
  int node_size = 0;
  fread(&node_size, sizeof(int), 1, fp);

  for (int i = 0; i < node_size; i++) {
    int name_len = 0;
    fread(&name_len, sizeof(int), 1, fp);

    char *name = (char *)malloc(sizeof(char) * name_len);
    fread(name, sizeof(char), name_len, fp);
    DLOG_INFO("params name: %s.", name);

    // Search all of the nodes in graph for finding the node that has the same name.
    // 加载结构时，需要检查重名的情况，确保到这里不会有重名node
    for (int j = 0; j < nodes_.size(); j++) {
      // 思考id号如何使用，预训练，有相同名字且不同id时，如何处理？
      // 改为只用名字来连接输入输出，加载完毕后，重新为各个node生成新的id号。
      // id号即对应这nodes_的下标索引，用于索引相应的node
      if (!strcmp(name, nodes_[j]->get_name().c_str())) {
        int size = 0;
        fread(&size, sizeof(int), 1, fp);

        if (size <= 1)
          break;

        const std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec = nodes_[j]->get_data_vec();
        for (int k = 1; k < size; k++) {
          int len = 0;
          fread(&len, sizeof(int), 1, fp);
          fread(data_vec[k]->GetPushCpuData(), sizeof(Dtype), len, fp);

          //float *tdata = (float *)data_vec[k]->GetPushCpuData();
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

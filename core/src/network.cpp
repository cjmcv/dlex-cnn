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
NetWork<Dtype>::NetWork(std::string name) {
  name_ = name;
  graph_.reset(new Graph<Dtype>());

  DLOG_INFO("NetWork constructed.");
}

template <typename Dtype>
NetWork<Dtype>::~NetWork() {
  DLOG_INFO("NetWork destructed.");
}

template <typename Dtype>
int NetWork<Dtype>::NetParamsInit() {
  graph_->ParamsInit();

  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::FeedDataByPrefetcher() {
  int ret = 0;
  std::pair < std::shared_ptr<Tensor<Dtype>>, std::shared_ptr<Tensor<Dtype>> > *batch;
  prefetcher_.FeedBatchOut(&batch);

  input_data_vec_.clear();
  input_data_vec_.push_back(batch->first);
  ret = graph_->SetInNode(input_data_vec_);

  label_data_vec_.clear();
  label_data_vec_.push_back(batch->second);
  ret += graph_->SetOutNode(label_data_vec_);

#ifdef USE_CUDA
  // Here is processed by the main thread, which is not the prefetcher one.
  // Ensure the copy is synchronous, so that the next batch is not copied in meanwhile.
  if (Task::mode() == tind::GPU)
    CUDA_DCHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
  prefetcher_.RefillBuffer(&batch);

  if (ret != 0)
    return -1;

  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::Forward(
  const std::shared_ptr<Tensor<Dtype>> input_data_tensor, 
  const std::shared_ptr<Tensor<Dtype>> label_data_tensor) {
  if (input_data_tensor == NULL) {
    int ret = FeedDataByPrefetcher();
    if (ret != 0)
      return -1;
  }
  else {
    int ret = 0;
    if (Task::mode() == tind::GPU) {
#ifdef USE_CUDA
      input_data_tensor->CheckPushGpuData();
#else
      DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif	
    }
    input_data_vec_.clear();
    input_data_vec_.push_back(input_data_tensor);
    ret = graph_->SetInNode(input_data_vec_);

    if (label_data_tensor != NULL) {
      if (Task::mode() == tind::GPU) {
#ifdef USE_CUDA
        label_data_tensor->CheckPushGpuData();
#else
        DLOG_ERR("CUDA programs are invalid, Please open the marco USE_CUDA");
#endif				
      }

      label_data_vec_.clear();
      label_data_vec_.push_back(label_data_tensor);
      ret += graph_->SetOutNode(label_data_vec_);
    }
    if (ret != 0)
      return -1;
  }

  graph_->ForwardGraph();
  //printf("finish Forward\n");
  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::Backward() {
  graph_->BackwardGraph();

  //update parameters
  const std::vector<std::shared_ptr<Node<Dtype>>> &nodes = graph_->get_graph_nodes();
  for (int i = 0; i < nodes.size(); i++) {
    std::string op_type = nodes[i]->get_inte_op()->get_op_type();
    if (!(op_type == "Input" || op_type == "Output")) {
      if (Task::mode() == tind::CPU)
        optimizer_->update(nodes[i]);
      else {
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
//Test only!

//Train phase may use this
template <typename Dtype>
std::shared_ptr<Tensor<Dtype>> NetWork<Dtype>::TestBatch(
  const std::shared_ptr<Tensor<Dtype>> input_data_tensor,
  const std::shared_ptr<Tensor<Dtype>> label_data_tensor) {
  ////set_phase(Phase::Test);
  Forward(input_data_tensor, label_data_tensor);
  //return lastOutput_[0];
  return NULL;
}

//////////////////////////////////////////////////////////////////////////
//Train only!
template <typename Dtype>
void NetWork<Dtype>::SetOptimizer(std::shared_ptr<Optimizer<Dtype>> optimizer) {
  this->optimizer_ = optimizer;
}

template <typename Dtype>
void NetWork<Dtype>::SetLearningRate(const float lr) {
  this->optimizer_->SetLearningRate(lr);
}

template <typename Dtype>
void NetWork<Dtype>::AddNode(const std::string &node_name,
  const std::vector<std::shared_ptr<Op<Dtype>>> &op,
  const std::vector<std::string> &in_node_names) {
  graph_->AddNode(node_name, op, in_node_names);
}

template <typename Dtype>
int NetWork<Dtype>::SwitchPhase(int phase) {
  this->phase_ = phase;
  //graph_->phase_ = phase;
  graph_->set_phase(phase);
  const std::vector<std::shared_ptr<Node<Dtype>>> &nodes = graph_->get_graph_nodes();
  for (int i = 0; i < nodes.size(); i++) {
    nodes[i]->set_phase(phase);
    nodes[i]->InferInteOp();	// get new op
    nodes[i]->InferOutShape();
    nodes[i]->InitOp();

    ////graph_->nodes_[i]->InitNode();	// 这里会改变node的权重
  }
  return 0;
}

template <typename Dtype>
inline int NetWork<Dtype>::SetIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names) {
  return graph_->SetIONodeName(in_node_names, out_node_names);
}

template <typename Dtype>
float NetWork<Dtype>::TrainBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor,
  const std::shared_ptr<Tensor<Dtype>> label_data_tensor) {
  //set_phase(Phase::Train);

  //printf("input_data_tensor[0] = %d\n", input_data_tensor->get_shape()[0]);
  //printf("TrainBatch start Forward\n");
  Forward(input_data_tensor, label_data_tensor);
  //printf("TrainBatch finish Forward\n");

  Dtype loss = 100.0;
  graph_->GetLoss("output", loss);

  //printf("TrainBatch start Backward\n");
  Backward();
  //printf("TrainBatch finish Backward\n");
  return loss;
}

template <typename Dtype>
int NetWork<Dtype>::GetNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &data) {
  graph_->GetNodeData(node_name, data);
  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::SaveBinModel(const std::string& model_file) {
  return true;
}

template <typename Dtype>
int NetWork<Dtype>::LoadBinModel(const std::string& model_file) {
  return true;
}

template <typename Dtype>
int NetWork<Dtype>::SaveStageModel(const std::string &path, const int stage) {
  std::string struct_file_name = "iter_" + std::to_string(stage) + ".struct";
  std::string param_file_name = "iter_" + std::to_string(stage) + ".param";

  FILE *st_fp = fopen(struct_file_name.c_str(), "w");
  graph_->WriteGraph2Text(st_fp);

  std::stringstream optss;
  optss << "optimizer:" << optimizer_->getOptName() << ",lr:" << optimizer_->getLearningRate() << ";";
  fprintf(st_fp, "%s\n", optss.str().c_str());

  fclose(st_fp);

  FILE *param_fp = fopen(param_file_name.c_str(), "wb");
  graph_->WriteGraphParam2Bin(param_fp);
  fclose(param_fp);

  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::ReadHyperParams(FILE *fp) {
  char cc[1000];	
  // Fetch optimizer's parameters
  while (EOF != fscanf(fp, "%s", cc)) {
    std::string cstr(cc);
    printf("read3: %s\n", cstr.c_str());

    std::string opt_str = FetchSubStr(cstr, "optimizer:", ",");
    float lr = atof(FetchSubStr(cstr, "lr:", ";").c_str());

    std::shared_ptr<dlex_cnn::Optimizer<Dtype>> optimizer;
    if (dlex_cnn::Optimizer<Dtype>::getOptimizerByStr(opt_str, optimizer)) {
      DLOG_ERR("[ NetWork::ReadHyperParams ]: Can not find optimizer by name - %s.", opt_str.c_str());
      return -1;
    }
    this->SetOptimizer(optimizer);

    if (lr > 0)
      this->SetLearningRate(lr);
    else {
      DLOG_ERR("[ NetWork::ReadHyperParams ]: Invalid learning rate -> ().", lr);
      return -1;
    }

    printf("read22_0: %s, %f\n", opt_str.c_str(), lr);
  }
  return 0;
}
template <typename Dtype>
int NetWork<Dtype>::LoadStageModel(const std::string &path, const int stage) {
  //ReadText2Graph(FILE *fp);
  std::string struct_file_name = "iter_" + std::to_string(stage) + ".struct";
  std::string param_file_name = "iter_" + std::to_string(stage) + ".param";

  FILE *st_fp = fopen(struct_file_name.c_str(), "r");
  graph_->ReadText2Graph(st_fp);
  ReadHyperParams(st_fp);

  fclose(st_fp);

  FILE *param_fp = fopen(param_file_name.c_str(), "rb");
  graph_->ReadBin2GraphParam(param_fp);
  fclose(param_fp);

  return 0;
}

template <typename Dtype>
int NetWork<Dtype>::NetWorkShow() {
  DLOG_INFO("***************************************************** ");
  DLOG_INFO("**************  Network's name: <%s>. *************\n", name_.c_str());
  DLOG_INFO("======================= Graph ======================= ");
  graph_->GraphShow();
  DLOG_INFO(">>>>>>>>>>>>>>>>>>>>> Optimizer <<<<<<<<<<<<<<<<<<<<< ");
  DLOG_INFO("lr: %f\n", optimizer_->getLearningRate());
  DLOG_INFO("***************************************************** ");
  return 0;
}
INSTANTIATE_CLASS(NetWork);

}//namespace

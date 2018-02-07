////////////////////////////////////////////////////////////////
// > Copyright (c) 2017, Jianming Chen. All rights reserved. 
// > https://github.com/cjmcv/cjmcv.github.io
// > file   network.h
// > brief  The Integration of each module
// > date   2017.07.18
////////////////////////////////////////////////////////////////

#ifndef DLEX_NETWORK_HPP_
#define DLEX_NETWORK_HPP_

#include <memory>
#include <vector>
#include "optimizer/optimizer.h"

#include "task.h"
#include "graph.h"
#include "node.h"
#include "tensor.h"
#include "io/data_prefetcher.h"

namespace dlex_cnn {
template <typename Dtype>
class NetWork {
public:
  NetWork(std::string name);
  virtual ~NetWork();

  int NetParamsInit();

  int SaveBinModel(const std::string &model_file);
  int LoadBinModel(const std::string &model_file);

  int SaveStageModel(const std::string &path, const int stage);
  int ReadHyperParams(FILE *fp);
  int LoadStageModel(const std::string &path, const int stage);

  std::shared_ptr<Tensor<Dtype>> TestBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor, const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);

  void SetOptimizer(std::shared_ptr<Optimizer<Dtype>> optimizer);
  void SetLearningRate(const float lr);

  inline int SetIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names);
  int FeedDataByPrefetcher();
  float TrainBatch(const std::shared_ptr<Tensor<Dtype>> input_data_tensor = NULL,
    const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);
  int GetNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &data);
  inline const std::shared_ptr<Graph<Dtype>> get_graph() { return graph_; };

  void AddNode(const std::string &node_name,
    const std::vector<std::shared_ptr<Op<Dtype>>> &op,
    const std::vector<std::string> &in_node_names = std::vector<std::string>());
  int SwitchPhase(int phase);

  // fill the input data and label date for training first, then compute graph Forward
  int Forward(const std::shared_ptr<Tensor<Dtype>> input_data_tensor = NULL, const std::shared_ptr<Tensor<Dtype>> label_data_tensor = NULL);
  // compute graph Backward and update nodes'paramaters
  int Backward();

  int NetWorkShow();

public:
  DataPrefetcher<Dtype> prefetcher_;

private:
  int device_id_;
  // Train/Test, should be the same as graph's
  int phase_ = tind::Train;
  std::string name_;

  // Mainly contains nodes and operators
  std::shared_ptr<Graph<Dtype>> graph_;
  // Optimizer to update node's paramater during training
  std::shared_ptr<Optimizer<Dtype>> optimizer_;

  // As the intermediate variable to transfer input/label data.
  std::vector<std::shared_ptr<Tensor<Dtype>>> input_data_vec_;
  std::vector<std::shared_ptr<Tensor<Dtype>>> label_data_vec_;
};

}
#endif DLEX_NETWORK_HPP_

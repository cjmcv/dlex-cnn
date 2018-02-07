////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node graph, support for node operation
//          including Forward and Backward
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_GRAPH_HPP_
#define DLEX_GRAPH_HPP_

//#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdlib.h>
#include <stack>

#include "util/timer.h"
#include "node.h"

namespace dlex_cnn {
template <typename Dtype>
class Graph {
public:
  explicit Graph();
  virtual ~Graph();

  int GetNodeIndex(const std::string &node_name, int &index);
  void AddNode(const std::string &node_name,
    const std::vector<std::shared_ptr<Op<Dtype>>> &op,
    const std::vector<std::string> &inNodeNames = std::vector<std::string>());

  void ParamsInit();
  int SetIONodeName(const std::vector<std::string> &in_node_names, const std::vector<std::string> &out_node_names);
  // Set input nodes of the graph.
  int SetInNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> inputData);
  // Set output nodes of the graph.
  int SetOutNode(const std::vector<std::shared_ptr<Tensor<Dtype>>> label_data);

  int ForwardGraph();
  int BackwardGraph();
  // Fetch loss that has been saved in one of the output nodes.
  int GetLoss(const std::string &node_name, Dtype &loss);
  // Fetch data in the specified node.
  int GetNodeData(const std::string &node_name, std::shared_ptr<Tensor<Dtype>> &data);

  int GraphShow();

  // Model Input and Output.
  int WriteGraph2Text(FILE *fp);
  int WriteGraph2Bin(FILE *fp);
  int WriteGraphParam2Bin(FILE *fp);

  int ReadText2Graph(FILE *fp);
  int ReadBin2Graph(FILE *fp);
  int ReadBin2GraphParam(FILE *fp);

  inline void set_phase(int phase) { phase_ = phase; };
  inline const std::vector<std::shared_ptr<Node<Dtype>>> &get_graph_nodes() { return nodes_; };

private:
  int phase_ = tind::Train;
  // The names of input nodes, and the order of vector's elements should matche the input data vector's.
  std::vector<std::string> in_node_names_;
  // The names of output nodes.
  std::vector<std::string> out_node_names_;
  // Bakckup in_node's name and idx
  std::map<std::string, int> in_nodes_map_;
  // Bakckup out_node's name and idx
  std::map<std::string, int> out_nodes_map_;
  // Bakckup names and idxs of all nodes in graph
  std::map<std::string, int> nodes_index_map_;
  // Nodes list
  std::vector<std::shared_ptr<Node<Dtype>>> nodes_;
  // Temporary stack for DFS in Forward and Backward
  std::stack<int> nodes_idx_stack_;
};
}
#endif //DLEX_GRAPH_HPP_

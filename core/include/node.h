////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node, the main component of Graph.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_NODE_HPP_
#define DLEX_NODE_HPP_

//#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>

//#include "dlex_datatype.h"
#include "tensor.h"
#include "operator/operator_base.h"
#include "operator/operator_hybrid.h"
#include "util/op_factory.h"

namespace dlex_cnn {
namespace tind {
  enum Phase { Train, Test };
}

template <typename Dtype>
class Node
{
public:
  explicit Node();
  virtual ~Node();

  // get members
  inline int get_phase() { return phase_; }
  inline int get_index() { return index_; }
  inline const std::string &get_name() { return name_; }
  inline const std::vector<int> &get_input_shape() { return input_shape_; }
  inline const std::vector<int> &get_output_shape() { return output_shape_; }
  inline const std::vector<int> &get_input_idx() { return inputs_index_; }
  inline const std::vector<std::string> &get_input_name() { return inputs_name_; }
  inline const std::vector<int> &get_output_idx() { return outputs_index_; }
  inline const std::vector<std::string> &get_output_name() { return outputs_name_; }
  inline const std::shared_ptr<Op<Dtype>> get_inte_op() { return inte_ops_; }
  inline const std::vector<std::shared_ptr<Tensor<Dtype>>> &get_data_vec() { return data_vec_; }

  inline const std::string GetOpParamBufStr() {
    op_param_str_ = inte_ops_->GenOpParamStr();
    return op_param_str_;
  }

  // set members
  inline void set_phase(int phase) { phase_ = phase; }
  inline void set_index(int index) { index_ = index; }
  inline void set_name(std::string name) { name_ = name; }
  inline void set_input_shape(std::vector<int> input_shape) { input_shape_ = input_shape; }
  inline void add_input_idx(int idx) { inputs_index_.push_back(idx); }
  inline void add_input_name(std::string name) { inputs_name_.push_back(name); }
  inline void add_output_idx(int idx) { outputs_index_.push_back(idx); }
  inline void add_output_name(std::string name) { outputs_name_.push_back(name); }
  inline void add_sub_ops(std::shared_ptr<Op<Dtype>> sub_op) { sub_ops_.push_back(sub_op); }
  inline void set_op_param_str(std::string str) { op_param_str_ = str; };

  // Allocate memory buffer for op, mainly includes diff_ and gradient_
  inline int InitOp() {
    int ret = inte_ops_->AllocOpBuf4Train(input_shape_, output_shape_);
    return ret;
  }

  // Allocate memory buffer for node according to inte_ops_
  inline int InitNode() {
    int ret = inte_ops_->AllocBuf4Node(input_shape_, output_shape_, data_vec_);
    return ret;
  }

  inline int InferOutShape() {
    int ret = inte_ops_->InferOutShape(input_shape_, output_shape_);
    return ret;
  }

  // Includes input_shape_, output_shape_ and the size of data_vec_
  int ResetDataSize(int index, const std::vector<int> &shape);

  // get the mapping relationship between a hybrid operation and serval operations
  int HybridOpMap(std::string &inte_op_type);

  // Infer and generate inte_ops_ on the basis of sub_ops_ and phase_
  int InferInteOp();

private:
  int phase_;

  // node index, for searching node in graph 
  int index_;
  // node name, as a unique symbol in the whole network
  std::string name_;

  std::vector<int> input_shape_;
  // output shape is equal to the size of this node data[0]
  std::vector<int> output_shape_;

  // sub-operators
  std::vector<std::shared_ptr<Op<Dtype>>> sub_ops_;

  // final operator for this node in this phase.
  // it will be one of the sub-operators or be assemed by some of those sub-operators
  std::shared_ptr<Op<Dtype>> inte_ops_;
  std::string op_param_str_ = "";

  std::vector<int> inputs_index_;
  std::vector<std::string> inputs_name_;
  std::vector<int> outputs_index_;
  std::vector<std::string> outputs_name_;

  // include in_data/weight/blas
  std::vector<std::shared_ptr<Tensor<Dtype>>> data_vec_;
  //std::vector<std::shared_ptr<Tensor<float>>> gradients_;	//include weight_gra/blas_gra

};
}
#endif //DLEX_NODE_HPP_
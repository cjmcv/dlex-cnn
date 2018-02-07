////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Node, the main component of Graph.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include <sstream>
#include "node.h"

namespace dlex_cnn {
template <typename Dtype>
Node<Dtype>::Node() {
  phase_ = tind::Train;

  inputs_index_.clear();	//是否需要？
  outputs_index_.clear();
}

template <typename Dtype>
Node<Dtype>::~Node() {}

template <typename Dtype>
int Node<Dtype>::HybridOpMap(std::string &inte_op_type) {
  int op_size = sub_ops_.size();
  if (op_size <= 0)
    return -1;

  if (op_size == 2) {
    std::string op_type_0 = sub_ops_[0]->get_op_type();
    std::string op_type_1 = sub_ops_[1]->get_op_type();
    for (int i = 0; i < OP_DOUBLE_NUM; i++) {
      if ((op_type_0 == kOpListDouble[i][1] && op_type_1 == kOpListDouble[i][2]) ||
        (op_type_1 == kOpListDouble[i][1] && op_type_0 == kOpListDouble[i][2])) {
        inte_op_type = kOpListDouble[i][0];
      }
    }
  }
  else if (op_size == 3) {
    //fill
  }
  else {
    DLOG_ERR("[ Node::HybridOpMap ]: sub_ops_.size() >= 4 that has not been implemented.");
    return -1;
  }

  return 0;
}

template <typename Dtype>
int Node<Dtype>::InferInteOp() {
  if (sub_ops_.size() <= 0) {
    DLOG_ERR("[ Node::InferInteOp ]: sub_ops_.size() <= 0.");
    return -1;
  }
  if (sub_ops_.size() == 1) {
    // 需要补充直接就是hop的情况
    inte_ops_ = sub_ops_[0];
    if (inte_ops_->get_op_category() == tind::eHybridOp) {

    }
  }
  else {
    std::string inte_op_str;
    HybridOpMap(inte_op_str);

    int s_index = -1;
    for (int i = 0; i < HOP_PHASEMAP_NUM; i++) {
      if (s_index != -1)
        break;
      if (inte_op_str == kHopPhaseMap[i][0])
        s_index = i;
    }
    if (s_index == -1) {
      DLOG_ERR("[ Node::InferInteOp ]: Can not find the hop with name < %s > in kHopPhaseMap.", inte_op_str);
      return -1;
    }
    //printf("inte_ops = %s\n", kHopPhaseMap[s_index][phase_ + 1].c_str());
    inte_ops_ = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType(kHopPhaseMap[s_index][phase_ + 1].c_str());
    if (inte_ops_->get_op_category() == tind::eHybridOp) {
      //printf("into inte_ops_->get_op_category() == tind::eHybridOp\n");
      dynamic_cast<dlex_cnn::HybridOp<Dtype> *>(inte_ops_.get())->SetSubOp(sub_ops_);
    }

    //dynamic_cast<dlex_cnn::InnerProductOp<float> *>(fc2.get())->SetOpParam(innerProductParam2);
  }

  return 0;
}

template <typename Dtype>
int Node<Dtype>::ResetDataSize(int index, const std::vector<int> &shape) {
  input_shape_ = shape;
  int ret = InferOutShape();
  if (ret == 0)
    data_vec_[index].reset(new Tensor<Dtype>(shape));
  return ret;
};

INSTANTIATE_CLASS(Node);
}
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/softmax_cross_entropy_hop.h"
#include <algorithm>
#include <sstream>

namespace dlex_cnn {
template <typename Dtype>
SoftmaxCrossEntropyLossHOp<Dtype>::SoftmaxCrossEntropyLossHOp() {
  op_type_ = "SoftmaxCrossEntropyLossH";
}

template <typename Dtype>
SoftmaxCrossEntropyLossHOp<Dtype>::SoftmaxCrossEntropyLossHOp(SoftmaxCrossEntropyLossHOpParam param) {
  op_type_ = "SoftmaxCrossEntropyLossH";
  param_ = param;
}

template <typename Dtype>
SoftmaxCrossEntropyLossHOp<Dtype>::~SoftmaxCrossEntropyLossHOp() {}

template <typename Dtype>
int SoftmaxCrossEntropyLossHOp<Dtype>::SetOpParam(const std::string &op_param_str) {
  // hop获取到hparam后，分别生成各子op对应的参数，对这些子op赋参
  return 0;
}

template <typename Dtype>
std::string SoftmaxCrossEntropyLossHOp<Dtype>::GenOpParamStr() const {
  std::stringstream param_str;
  param_str << ",";
  return param_str.str();
}
template <typename Dtype>
int SoftmaxCrossEntropyLossHOp<Dtype>::InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) {
  out_shape = in_shape;
  return 0;
}

template <typename Dtype>
int SoftmaxCrossEntropyLossHOp<Dtype>::AllocBuf4Node(const std::vector<int> &in_shape,
  const std::vector<int> &out_shape,
  std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::AllocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  data.clear();
  data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  return 0;
}

template <typename Dtype>
int SoftmaxCrossEntropyLossHOp<Dtype>::AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::AllocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  diff_.clear();
  diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  if (HybridOp<Dtype>::sub_ops_.size() == 0) {
    // can set sub_ops's params by loading inteOp's params
    std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s
      = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Softmax");
    //dlex_cnn::SoftmaxOpParam softmaxParam;
    //dynamic_cast<dlex_cnn::SoftmaxOp<Dtype> *>(sm_s.get())->SetOpParam(softmaxParam);

    std::shared_ptr<dlex_cnn::Op<Dtype>> cel_s
      = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("CrossEntropyLoss");
    //dlex_cnn::CrossEntropyLossOpParam CELParam;
    //dynamic_cast<dlex_cnn::CrossEntropyLossOp<Dtype> *>(cel_s.get())->SetOpParam(CELParam);

    HybridOp<Dtype>::sub_ops_.push_back(sm_s);
    HybridOp<Dtype>::sub_ops_.push_back(cel_s);
  }

  return 0;
}

template <typename Dtype>
void SoftmaxCrossEntropyLossHOp<Dtype>::Forward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  //printf("start SoftmaxCrossEntropyLossHOp\n");
  int op_ind[2] = { -1 };
  bool flag = false;
  if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "Softmax"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "CrossEntropyLoss") {
    op_ind[0] = 0;
    op_ind[1] = 1;
    flag = true;
  }
  else if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "CrossEntropyLoss"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "Softmax") {
    op_ind[0] = 1;
    op_ind[1] = 0;
    flag = true;
  }

  if (flag == false) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]: the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
    return;
  }

  HybridOp<Dtype>::sub_ops_[op_ind[0]]->Forward(prev, next);
  HybridOp<Dtype>::sub_ops_[op_ind[1]]->Forward(next, next);
  //printf("finish SoftmaxCrossEntropyLossHOp\n");
}

template <typename Dtype>
void SoftmaxCrossEntropyLossHOp<Dtype>::Backward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {
  int op_ind[2] = { -1 };
  bool flag = false;
  if (HybridOp<Dtype>::sub_ops_.size() != 2) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]:  sub_ops_.size() != 2 \n");
    return;
  }
  if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "Softmax"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "CrossEntropyLoss") {
    op_ind[0] = 0;
    op_ind[1] = 1;
    flag = true;
  }
  else if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "CrossEntropyLoss"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "Softmax") {
    op_ind[0] = 1;
    op_ind[1] = 0;
    flag = true;
  }
  if (flag == false) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]:  the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
    return;
  }
  //lastOutput in next[0], lastDiff will be generated in next_diff
  HybridOp<Dtype>::sub_ops_[op_ind[1]]->Backward(prev, next, next_diff, next_diff);
  HybridOp<Dtype>::sub_ops_[op_ind[0]]->Backward(prev, next, prev_diff, next_diff);
}

#ifdef USE_CUDA
template <typename Dtype>
void SoftmaxCrossEntropyLossHOp<Dtype>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  int op_ind[2] = { -1 };
  bool flag = false;
  if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "Softmax"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "CrossEntropyLoss") {
    op_ind[0] = 0;
    op_ind[1] = 1;
    flag = true;
  }
  else if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "CrossEntropyLoss"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "Softmax") {
    op_ind[0] = 1;
    op_ind[1] = 0;
    flag = true;
  }

  if (flag == false) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]: the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
    return;
  }

  HybridOp<Dtype>::sub_ops_[op_ind[0]]->Forward_gpu(prev, next);
  HybridOp<Dtype>::sub_ops_[op_ind[1]]->Forward_gpu(next, next);
}

template <typename Dtype>
void SoftmaxCrossEntropyLossHOp<Dtype>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {
  int op_ind[2] = { -1 };
  bool flag = false;
  if (HybridOp<Dtype>::sub_ops_.size() != 2) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]:  sub_ops_.size() != 2 \n");
    return;
  }
  if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "Softmax"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "CrossEntropyLoss") {
    op_ind[0] = 0;
    op_ind[1] = 1;
    flag = true;
  }
  else if (HybridOp<Dtype>::sub_ops_[0]->get_op_type() == "CrossEntropyLoss"
    && HybridOp<Dtype>::sub_ops_[1]->get_op_type() == "Softmax") {
    op_ind[0] = 1;
    op_ind[1] = 0;
    flag = true;
  }
  if (flag == false) {
    DLOG_ERR("[ SoftmaxCrossEntropyLossHOp::Forward ]:  the type of sub_ops_ are not (Softmax + CrossEntropyLoss) \n");
    return;
  }
  //lastOutput in next[0], lastDiff will be generated in next_diff
  HybridOp<Dtype>::sub_ops_[op_ind[1]]->Backward_gpu(prev, next, next_diff, next_diff);
  HybridOp<Dtype>::sub_ops_[op_ind[0]]->Backward_gpu(prev, next, prev_diff, next_diff);
}
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossHOp);

}//namespace

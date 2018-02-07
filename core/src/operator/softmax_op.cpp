////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Softmax.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/softmax_op.h"
#include <algorithm>
#include <sstream>

namespace dlex_cnn {
template <typename Dtype>
SoftmaxOp<Dtype>::SoftmaxOp() {
  op_type_ = "Softmax";
}

template <typename Dtype>
SoftmaxOp<Dtype>::SoftmaxOp(SoftmaxOpParam param) {
  op_type_ = "Softmax";
  param_ = param;
}

template <typename Dtype>
SoftmaxOp<Dtype>::~SoftmaxOp() {}

template <typename Dtype>
int SoftmaxOp<Dtype>::SetOpParam(const std::string &op_param_str) {
  return 0;
}

template <typename Dtype>
std::string SoftmaxOp<Dtype>::GenOpParamStr() const {
  std::stringstream param_str;
  param_str << ",";
  return param_str.str();
}

template <typename Dtype>
int SoftmaxOp<Dtype>::InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) {
  out_shape = in_shape;
  return 0;
}

template <typename Dtype>
int SoftmaxOp<Dtype>::AllocBuf4Node(const std::vector<int> &in_shape,
  const std::vector<int> &out_shape,
  std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ SoftmaxOp::AllocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  data.clear();
  data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  return 0;
}

template <typename Dtype>
int SoftmaxOp<Dtype>::AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ SoftmaxOp::AllocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  diff_.clear();
  diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  return 0;
}

template <typename Dtype>
void SoftmaxOp<Dtype>::Forward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  const std::vector<int> prev_data_size = prev[0]->get_size();
  const std::vector<int> next_data_size = next[0]->get_size();
  //const std::vector<int> prev_data_shape = prev[0]->get_shape();
  const std::vector<int> next_data_shape = next[0]->get_shape();

  Dtype *prev_data_base = (Dtype *)prev[0]->GetPushCpuData();
  Dtype *next_data_base = (Dtype *)next[0]->GetCpuData();

  const int next_data_num = next_data_shape[tind::eNum];
  const int prev_data_size3D = prev_data_size[tind::e3D];
  const int next_data_size3D = next_data_size[tind::e3D];

  next[0]->SetCpuZero();
  for (int nn = 0; nn < next_data_num; nn++) {
    const Dtype* prev_data = prev_data_base + nn * prev_data_size3D;
    Dtype* next_data = next_data_base + nn * next_data_size3D;

    //step1 : find max value
    Dtype maxVal = prev_data[0];
    for (int prevDataIdx = 0; prevDataIdx < prev_data_size3D; prevDataIdx++) {
      maxVal = std::max(maxVal, prev_data[prevDataIdx]);
    }
    //step2 : sum
    Dtype sum = 0;
    for (int prevDataIdx = 0; prevDataIdx < prev_data_size3D; prevDataIdx++) {
      next_data[prevDataIdx] = std::exp(prev_data[prevDataIdx] - maxVal);
      sum += next_data[prevDataIdx];
    }
    //step3 : div
    for (int prevDataIdx = 0; prevDataIdx < prev_data_size3D; prevDataIdx++) {
      next_data[prevDataIdx] = next_data[prevDataIdx] / sum;
    }
  }
}

template <typename Dtype>
void SoftmaxOp<Dtype>::Backward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {

  const std::vector<int> prev_data_size = prev[0]->get_size();
  const std::vector<int> next_data_size = next[0]->get_size();
  const std::vector<int> prev_diff_size = prev_diff[0]->get_size();
  const std::vector<int> next_diff_size = next_diff[0]->get_size();

  const std::vector<int> prev_data_shape = prev[0]->get_shape();
  const std::vector<int> next_data_shape = next[0]->get_shape();
  const std::vector<int> prev_diff_shape = prev_diff[0]->get_shape();
  const std::vector<int> next_diff_shape = next_diff[0]->get_shape();

  Dtype *prev_data_base = (Dtype *)prev[0]->GetPushCpuData();
  Dtype *next_data_base = (Dtype *)next[0]->GetPushCpuData();
  Dtype *prev_diff_base = (Dtype *)prev_diff[0]->GetCpuData();
  Dtype *next_diff_base = (Dtype *)next_diff[0]->GetPushCpuData();

  if (prev_data_size[tind::e4D] != next_data_size[tind::e4D]) {
    DLOG_ERR("[ SoftmaxOp::Backward ]: the size of input and output data must be equal \n");
    return;
  }
  if (prev_diff_size[tind::e4D] != next_diff_size[tind::e4D]) {
    DLOG_ERR("[ SoftmaxOp::Backward ]: the size of input diff and output diff must be equal \n");
    return;
  }
  if (prev_diff_size[tind::e4D] != prev_data_size[tind::e4D]) {
    DLOG_ERR("[ SoftmaxOp::Backward ]: the size of input diff and output data must be equal \n");
    return;
  }

  //update prev_diff
  prev_diff[0]->SetCpuZero();
  const int prev_data_size3D = prev_data_size[tind::e3D];
  const int next_data_size3D = next_data_size[tind::e3D];
  const int prev_diff_size3D = prev_diff_size[tind::e3D];
  const int next_diff_size3D = next_diff_size[tind::e3D];
  for (int pn = 0; pn < prev_data_shape[tind::eNum]; pn++) {
    const Dtype* prev_data = prev_data_base + pn * prev_data_size3D;
    const Dtype* next_data = next_data_base + pn * next_data_size3D;
    const Dtype* next_diff = next_diff_base + pn * next_diff_size3D;
    Dtype* prev_diff = prev_diff_base + pn * prev_diff_size3D;

    for (int prev_diff_idx = 0; prev_diff_idx < prev_diff_size3D; prev_diff_idx++) {
      const Dtype val_next_data = next_data[prev_diff_idx];
      Dtype val_prev_diff = prev_diff[prev_diff_idx];
      for (int next_diff_idx = 0; next_diff_idx < next_diff_size3D; next_diff_idx++) {
        val_prev_diff -= val_next_data * next_data[next_diff_idx] * next_diff[next_diff_idx];
      }
      prev_diff[prev_diff_idx] = val_prev_diff;
    }

    for (int idx = 0; idx < prev_diff_size3D; idx++)
      prev_diff[idx] += next_data[idx] * next_diff[idx];
  }
}

INSTANTIATE_CLASS(SoftmaxOp);

}//namespace
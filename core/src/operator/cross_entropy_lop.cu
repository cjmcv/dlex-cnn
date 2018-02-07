////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  CrossEntropyLossOp. Only for softmax for now.
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include "operator/cross_entropy_lop.h"
#include <sstream>

namespace dlex_cnn {
template <typename Dtype>
void CrossEntropyLossOp<Dtype>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  Forward(prev, next);
}

template <typename Dtype>
__global__ void CrossEntropyBackwardKernel(const int num,
  const int labels_size3D, Dtype* label_data_base,
  const int output_size3D, Dtype* output_data_base,
  const int diff_size3D, Dtype* prev_diff_base) {
  CUDA_KERNEL_LOOP(index, num) {
    const Dtype* label_data = label_data_base + index * labels_size3D;
    const Dtype* output_data = output_data_base + index * output_size3D;
    Dtype* diff_data = prev_diff_base + index * diff_size3D;
    for (int idx = 0; idx < diff_size3D; idx++) {
      diff_data[idx] -= ((label_data[idx] / (output_data[idx])));
    }
  }
}

template <typename Dtype>
void CrossEntropyLossOp<Dtype>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff
  ) {
  const int output_size4D = next[0]->get_size()[tind::e4D];
  if (labels_ == NULL || labels_->get_size()[tind::e4D] != output_size4D) {
    DLOG_ERR("[ CrossEntropyLossOp::Backward ]: labels_ is invalid \n");
    return;
  }

  Dtype* label_data = (Dtype *)labels_->GetPushGpuData();

  const int labels_size3D = labels_->get_size()[tind::e3D];
  const int output_size3D = next[0]->get_size()[tind::e3D];
  const int diff_size3D = prev_diff[0]->get_size()[tind::e3D];

  Dtype* label_data_base = (Dtype *)labels_->GetPushGpuData();
  Dtype* output_data_base = (Dtype *)next[0]->GetPushGpuData();
  Dtype* prev_diff_base = (Dtype *)prev_diff[0]->GetGpuData();

  prev_diff[0]->SetGpuZero();

  int num = next[0]->get_shape()[tind::eNum];
  CrossEntropyBackwardKernel<Dtype> << <DLEX_GET_BLOCKS(num), DLEX_CUDA_NUM_THREADS >> >(
    num, labels_size3D, label_data_base,
    output_size3D, output_data_base,
    diff_size3D, prev_diff_base);

  CUDA_POST_KERNEL_CHECK;
}

template void CrossEntropyLossOp<float>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<float>>> &prev,
  const std::vector<std::shared_ptr<Tensor<float>>> &next);
template void CrossEntropyLossOp<double>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<double>>> &prev,
  const std::vector<std::shared_ptr<Tensor<double>>> &next);
template void CrossEntropyLossOp<float>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<float>>> &prev,
  const std::vector<std::shared_ptr<Tensor<float>>> &next,
  const std::vector<std::shared_ptr<Tensor<float>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<float>>> &next_diff);
template void CrossEntropyLossOp<double>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<double>>> &prev,
  const std::vector<std::shared_ptr<Tensor<double>>> &next,
  const std::vector<std::shared_ptr<Tensor<double>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<double>>> &next_diff);
}//namespace
#endif

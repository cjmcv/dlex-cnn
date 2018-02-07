////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Simple activation functions.
// > author Jianming Chen
////////////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include "operator/activation_simple_op.h"
#include <sstream>

namespace dlex_cnn {
//inline Dtype relu(Dtype x) { return std::max(x, Dtype(0)) + param_.negative_slope * std::min(x, Dtype(0)); }
template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out, float negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

//inline Dtype sigmoid(Dtype x) { return 1. / (1. + exp(-x)); }
template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

//inline Dtype tanh(Dtype x) { return std::tanh(x); }
template <typename Dtype>
__global__ void TanHForward(const int n, const Dtype* prev_data, Dtype* next_data) {
  CUDA_KERNEL_LOOP(index, n) {
    next_data[index] = tanh(prev_data[index]);
  }
}

template <typename Dtype>
void ActivationOp<Dtype>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  const int prev_size3D = prev[0]->get_size()[tind::e3D];
  const int next_size3D = next[0]->get_size()[tind::e3D];
  Dtype* prev_data = (Dtype *)prev[0]->GetPushGpuData();
  Dtype* next_data = (Dtype *)next[0]->GetGpuData();

  next[0]->SetGpuZero();
  for (int n = 0; n < prev[0]->get_shape()[tind::eNum]; n++) {
    Dtype* prev_data_n = prev_data + n * prev_size3D;
    Dtype* next_data_n = next_data + n * next_size3D;
    switch (param_.activation_type) {
    case tind::Activation::eReLU:
      ReLUForward<Dtype> << <DLEX_GET_BLOCKS(prev_size3D), DLEX_CUDA_NUM_THREADS >> >(
        prev_size3D, prev_data_n, next_data_n, param_.negative_slope);
      break;
    case tind::Activation::eSigmoid:
      SigmoidForward<Dtype> << <DLEX_GET_BLOCKS(prev_size3D), DLEX_CUDA_NUM_THREADS >> >(
        prev_size3D, prev_data_n, next_data_n);
      break;
    case tind::Activation::eTanh:
      TanHForward<Dtype> << <DLEX_GET_BLOCKS(prev_size3D), DLEX_CUDA_NUM_THREADS >> >(
        prev_size3D, prev_data_n, next_data_n);
      break;
    default:
      DLOG_ERR("Unknown activation type.");
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

// inline Dtype rev_relu(Dtype px, Dtype diff_next) 
// { return diff_next * ((px > 0) + param_.negative_slope * (px <= 0)); }
template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* prev_data,
  const Dtype* next_diff, Dtype* prev_diff, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype px = prev_data[index];
    prev_diff[index] = next_diff[index] * ((px > 0) + negative_slope * (px <= 0));
  }
}

// inline Dtype rev_sigmoid(Dtype nx, Dtype diff_next) { return diff_next * nx * (1. - nx); }
template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* next_data,
  const Dtype* next_diff, Dtype* prev_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype nx = next_data[index];
    prev_diff[index] = next_diff[index] * nx * (1. - nx);
  }
}

// inline Dtype rev_tanh(Dtype nx, Dtype diff_next) { return diff_next * (1 - nx * nx); }
template <typename Dtype>
__global__ void TanHBackward(const int n, const Dtype* next_data,
  const Dtype* next_diff, Dtype* prev_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype nx = next_data[index];
    prev_diff[index] = next_diff[index] * (1 - nx * nx);
  }
}

template <typename Dtype>
void ActivationOp<Dtype>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff
  ) {
  const int prev_size3D = prev[0]->get_size()[tind::e3D];
  const int next_size3D = next[0]->get_size()[tind::e3D];
  const int prev_diff_size3D = prev_diff[0]->get_size()[tind::e3D];
  const int next_diff_size3D = next_diff[0]->get_size()[tind::e3D];
  Dtype* prev_data = (Dtype *)prev[0]->GetPushGpuData();
  Dtype* next_data = (Dtype *)next[0]->GetPushGpuData();
  Dtype* prev_diff_data = (Dtype *)prev_diff[0]->GetGpuData();
  Dtype* next_diff_data = (Dtype *)next_diff[0]->GetPushGpuData();

  prev_diff[0]->SetGpuZero();
  for (int n = 0; n < prev[0]->get_shape()[tind::eNum]; n++) {
    Dtype* prev_data_n = prev_data + n * prev_size3D;
    Dtype* next_data_n = next_data + n * next_size3D;
    Dtype* prev_diff_data_n = prev_diff_data + n * prev_diff_size3D;
    Dtype* next_diff_data_n = next_diff_data + n * next_diff_size3D;

    switch (param_.activation_type) {
    case tind::Activation::eReLU:
      ReLUBackward<Dtype> << <DLEX_GET_BLOCKS(prev_size3D), DLEX_CUDA_NUM_THREADS >> >(
        prev_size3D, prev_data_n, next_diff_data_n, prev_diff_data_n, param_.negative_slope);
      break;
    case tind::Activation::eSigmoid:
      SigmoidBackward<Dtype> << <DLEX_GET_BLOCKS(next_size3D), DLEX_CUDA_NUM_THREADS >> >(
        next_size3D, next_data_n, next_diff_data_n, prev_diff_data_n);
      break;
    case tind::Activation::eTanh:
      TanHBackward<Dtype> << <DLEX_GET_BLOCKS(next_size3D), DLEX_CUDA_NUM_THREADS >> >(
        next_size3D, next_data_n, next_diff_data_n, prev_diff_data_n);
      break;
    default:
      DLOG_ERR("Unknown activation type.");
    }
  }
  CUDA_POST_KERNEL_CHECK;
}
template void ActivationOp<float>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<float>>> &prev,
  const std::vector<std::shared_ptr<Tensor<float>>> &next);
template void ActivationOp<double>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<double>>> &prev,
  const std::vector<std::shared_ptr<Tensor<double>>> &next);
template void ActivationOp<float>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<float>>> &prev,
  const std::vector<std::shared_ptr<Tensor<float>>> &next,
  const std::vector<std::shared_ptr<Tensor<float>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<float>>> &next_diff);
template void ActivationOp<double>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<double>>> &prev,
  const std::vector<std::shared_ptr<Tensor<double>>> &next,
  const std::vector<std::shared_ptr<Tensor<double>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<double>>> &next_diff);
}//namespace
#endif
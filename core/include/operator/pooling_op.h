////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_POOLING_HPP_
#define DLEX_OP_POOLING_HPP_

#include "operator_base.h"
#include "util/math_functions.h"
#include "tensor.h"

#ifdef USE_OP_TEST
#include "../../core_test/operator_test/pooling_op_test.h"
#endif


namespace dlex_cnn {
namespace tind {
  enum PoolingType { eMAX, eAVE, eSTOCHASTIC };
}
struct PoolingOpParam : public OpParam {
  tind::PoolingType pooling_type = tind::eAVE;
  int kernel_h = 3, kernel_w = 3;
  int stride_h = 1, stride_w = 1;
  int pad_h = 0, pad_w = 0;
  //int channels;
  //int height_, width_;					// Input dimension (prev)
  //int pooled_height_, pooled_width_;	// Input dimension(after pooling, next)
  bool global_pooling = false;			// Kernel size is equal to Input's if it is ture;
};

template <typename Dtype>
class PoolingOp : public Op<Dtype> {
#ifdef USE_OP_TEST
  template <typename T>
  friend class PoolingOpTest;
#endif
public:
  PoolingOp();
  PoolingOp(PoolingOpParam param);
  virtual ~PoolingOp();
  inline virtual int SetOpParam(PoolingOpParam op_param) { param_ = op_param; return 0; };
  virtual int SetOpParam(const std::string &op_param_str) override;

private:
  inline virtual const std::string &get_op_type() const override { return op_type_; };
  inline virtual const int get_op_category() const override { return tind::eNormOp; };
  inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &get_op_diff() override { return diff_; };

  virtual std::string GenOpParamStr() const override;
  virtual int InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) override;

  virtual int AllocBuf4Node(const std::vector<int> &in_shape,
    const std::vector<int> &out_shape,
    std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
  virtual int AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) override;

  // Refer to Caffe
  // In max pooling, it should mark the location(index) of max number in each sliding window.
  // In average pooling, just get the mean value;
  virtual void Forward(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
  // In max pooling, According to the mask we have got in Forward, we just only update the points which is the max of each sliding windows in Forward.
  // In average pooling, each next value will be divided by kernel size, and fill the relevant sliding window with the quotient;
  // If sliding windows are overlap, the output will be accumulated.
  virtual void Backward(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) override;
#ifdef USE_CUDA
  virtual void Forward_gpu(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
  virtual void Backward_gpu(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) override;
#endif

private:
  std::string op_type_;
  std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
  std::shared_ptr<Tensor<Dtype>> rand_idx_map_;
  std::shared_ptr<Tensor<int>> max_idx_map_;
  PoolingOpParam param_;

};

/////////////////////////////  Creator   /////////////////////////////////

template <typename Dtype>
PoolingOp<Dtype> *CreateOp(PoolingOpParam &param){
  PoolingOp<Dtype> *op = NULL;
  op = new PoolingOp<Dtype>(param);
  return op;
}
template <typename Dtype>
std::shared_ptr<Op<Dtype>> CreatePoolingOp() {
  return std::shared_ptr<Op<Dtype>>(new PoolingOp<Dtype>());
}
}
#endif

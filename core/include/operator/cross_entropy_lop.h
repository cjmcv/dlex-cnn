////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_CROSS_ENTROPY_HPP_
#define DLEX_OP_CROSS_ENTROPY_HPP_

#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn {
struct CrossEntropyLossOpParam : public OpParam {};

template <typename Dtype>
class CrossEntropyLossOp : public Op<Dtype> {
public:
  CrossEntropyLossOp();
  CrossEntropyLossOp(CrossEntropyLossOpParam param);
  virtual ~CrossEntropyLossOp();
  inline virtual int SetOpParam(CrossEntropyLossOpParam op_param) { param_ = op_param; return 0; };
  virtual int SetOpParam(const std::string &op_param_str) override;

private:
  inline virtual const std::string &get_op_type() const override { return op_type_; };
  inline virtual const int get_op_category() const override { return tind::eLossOp; };
  inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &get_op_gradient() override { return gradient_; };
  inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &get_op_diff() override { return diff_; };

  virtual std::string GenOpParamStr() const override;
  virtual int InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) override;
  //virtual int solveInnerParams(const std::vector<int> &in_shape, const std::vector<int> &out_shape,
  //	std::vector<std::shared_ptr<Tensor<Dtype>>> &data) override;
  virtual int AllocBuf4Node(const std::vector<int> &in_shape,
    const std::vector<int> &out_shape,
    std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
  virtual int AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) override;

  // Different loss definition for different activation function: http://m.blog.csdn.net/u012436149/article/details/69660214
  //
  // prev[0] -> outdata of the last node (activation node).
  // next[1] -> org_label_data, has been fetched from label file, will be converted to fit this loss operation.
  // next[2] -> ouput loss.
  virtual void Forward(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;

  // prev_diff[0] -> gradients for Backward.
  // next[0] -> outdata of the last node.
  // prev and next_diff is inactive in this function.
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

  std::vector<std::shared_ptr<Tensor<Dtype>>> gradient_;
  std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;

  CrossEntropyLossOpParam param_;

  // Have the matched dimensions to this loss operator
  // Will be converted by the labels data in outNode
  std::shared_ptr<Tensor<Dtype>> labels_;
};

/////////////////////////////  Creator   /////////////////////////////////
template <typename Dtype>
CrossEntropyLossOp<Dtype> *CreateOp(CrossEntropyLossOpParam &param) {
  CrossEntropyLossOp<Dtype> *op = NULL;
  op = new CrossEntropyLossOp<Dtype>(param);
  return op;
}
template <typename Dtype>
std::shared_ptr<Op<Dtype>> CreateCrossEntropyLossOp() {
  return std::shared_ptr<Op<Dtype>>(new CrossEntropyLossOp<Dtype>());
}
}
#endif

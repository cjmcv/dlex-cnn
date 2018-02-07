////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_INNER_PRODUCT_HPP_
#define DLEX_OP_INNER_PRODUCT_HPP_

#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn {
struct InnerProductOpParam : public OpParam {
  bool blas_enable;
  int num_hidden;
};

template <typename Dtype>
class InnerProductOp : public Op<Dtype> {
public:
  InnerProductOp();
  InnerProductOp(InnerProductOpParam param);
  virtual ~InnerProductOp();
  inline virtual int SetOpParam(InnerProductOpParam op_param) { param_ = op_param; return 0; };
  virtual int SetOpParam(const std::string &op_param_str) override;

private:
  inline virtual const std::string &get_op_type() const override { return op_type_; };
  inline virtual const int get_op_category() const override { return tind::eNormOp; };
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

  // fc: o1 = i1 x w11 + i2 x w12 + i3 x w13 + b1;
  //     o2 = i1 x w21 + i2 x w22 + i3 x w23 + b2;
  //     o3 = i1 x w31 + i2 x w32 + i3 x w33 + b3;
  // ->  matrix O = I * W + Blas  -> can use GEMM to implement
  //
  // input(num, channels, height, width) as matrix I(M,K) = I(num, channels*height*width),
  // weight as matrix W(N, K) = W(hidden_num, channels*height*width),
  // blas as matrix B(N) = B(hidden_num),
  // output(num, hidden_num, 1, 1) as matrix O(M,N) = O(num, hidden_num),
  // ->  O = I * W' + B. (Our case).
  //
  // note: In caffe, there has a variable "transpose_". 
  //		 If the variable is "true", assume transposed weights, that means weights have been saved like W(K,N). The formula should be O = I * W + B.
  //       Or if it is "false", then weights have been saved like W(K,N), and the formula will be changed to O = I * W' + B.
  virtual void Forward(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;

  // 1¡¢Update prev_diff, according to next_diff and weight.
  // prev_diff(num, in3DSize) = next_diff(num, hidden_num) * weight(hidden_num, in3DSize).
  // -> prev_diff(num, prev_diff_size[tind::e3D]) = next_diff(num, next_diff_size[tind::e3D]) * weight(next_diff_size[tind::e3D], in3DSize).
  //
  // 2¡¢Get weight gradient.
  // next_diff(num, hidden_num) -> next_diff'(hidden_num, num).
  // O(M,N) = weightGradient(hidden_num, in3DSize) = next_diff'(hidden_num, num) * prev_data(num, in3DSize).
  // -> M=hidden_num, N=in3DSize, K=num.
  //
  // 3¡¢update bias, just accumulate next_diff_data.
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
  InnerProductOpParam param_;

};

/////////////////////////////  Creator   /////////////////////////////////

template <typename Dtype>
InnerProductOp<Dtype> *CreateOp(InnerProductOpParam &param) {
  InnerProductOp<Dtype> *op = NULL;
  op = new InnerProductOp<Dtype>(param);
  return op;
}
template <typename Dtype>
std::shared_ptr<Op<Dtype>> CreateInnerProductOp() {
  return std::shared_ptr<Op<Dtype>>(new InnerProductOp<Dtype>());
}
}
#endif

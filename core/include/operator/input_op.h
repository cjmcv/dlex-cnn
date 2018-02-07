////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_INPUT_HPP_
#define DLEX_OP_INPUT_HPP_

#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn {
struct InputOpParam : public OpParam {
  InputOpParam() {};
  InputOpParam(int in_num, int in_channels, int in_height, int in_width) {
    num = in_num;
    channels = in_channels;
    height = in_height;
    width = in_width;
  }
  int num;
  int channels;
  int height;
  int width;
};

template <typename Dtype>
class InputOp : public Op<Dtype> {
  //FRIEND_WITH_NETWORK
public:
  InputOp();
  InputOp(InputOpParam param);
  virtual ~InputOp();
  inline virtual int SetOpParam(InputOpParam op_param) { param_ = op_param; return 0; };
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

  virtual void Forward(
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
    const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) override;
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
  InputOpParam param_;
  //std::vector<std::shared_ptr<Tensor<float>>> gradient_;
  std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
};

/////////////////////////////  Creator   /////////////////////////////////

template <typename Dtype>
InputOp<Dtype> *CreateOp(InputOpParam &param) {
  InputOp<Dtype> *op = NULL;
  op = new InputOp<Dtype>(param);
  return op;
}
template <typename Dtype>
std::shared_ptr<Op<Dtype>> CreateInputOp() {
  return std::shared_ptr<Op<Dtype>>(new InputOp<Dtype>());
}
}
#endif

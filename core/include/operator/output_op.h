////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_OP_OUTPUT_HPP_
#define DLEX_OP_OUTPUT_HPP_

#include "operator_base.h"
#include "tensor.h"

namespace dlex_cnn {
struct OutputOpParam : public OpParam {
  int label_dim;
};

template <typename Dtype>
class OutputOp : public Op<Dtype> {
  //FRIEND_WITH_NETWORK
public:
  OutputOp();
  OutputOp(OutputOpParam param);
  virtual ~OutputOp();
  inline virtual int SetOpParam(OutputOpParam op_param) { param_ = op_param; return 0; };
  virtual int SetOpParam(const std::string &op_param_str) override;

private:
  //DECLARE_LAYER_TYPE;
  inline virtual const std::string &get_op_type() const override { return op_type_; };
  inline virtual const int get_op_category() const override { return tind::eNormOp; };
  inline virtual std::vector<std::shared_ptr<Tensor<Dtype>>> &get_op_diff() override { return diff_; };

  virtual std::string GenOpParamStr() const override;
  virtual int InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) override;
  virtual int AllocBuf4Node(const std::vector<int> &in_shape,
    const std::vector<int> &out_shape,
    std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const override;
  virtual int AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) override;

private:
  std::string op_type_;
  OutputOpParam param_;
  //std::vector<std::shared_ptr<Tensor<float>>> gradient_;
  std::vector<std::shared_ptr<Tensor<Dtype>>> diff_;
};

/////////////////////////////  Creator   /////////////////////////////////

template <typename Dtype>
OutputOp<Dtype> *CreateOp(OutputOpParam &param) {
  OutputOp<Dtype> *op = NULL;
  op = new OutputOp<Dtype>(param);
  return op;
}
template <typename Dtype>
std::shared_ptr<Op<Dtype>> CreateOutputOp(){
  return std::shared_ptr<Op<Dtype>>(new OutputOp<Dtype>());
}
}
#endif

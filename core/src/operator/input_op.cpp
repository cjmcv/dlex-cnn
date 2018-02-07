////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/input_op.h"
#include <sstream>

namespace dlex_cnn {	
template <typename Dtype>
InputOp<Dtype>::InputOp() {
  op_type_ = "Input";
}

template <typename Dtype>
InputOp<Dtype>::InputOp(InputOpParam param) {
  op_type_ = "Input";
  param_ = param;
}

template <typename Dtype>
InputOp<Dtype>::~InputOp() {}

template <typename Dtype>
int InputOp<Dtype>::SetOpParam(const std::string &op_param_str) {
  std::string opt_str = op_param_str;
  param_.num = atoi(FetchSubStr(opt_str, "num:", ",").c_str());
  param_.channels = atoi(FetchSubStr(opt_str, "channels:", ",").c_str());
  param_.height = atoi(FetchSubStr(opt_str, "height:", ",").c_str());
  param_.width = atoi(FetchSubStr(opt_str, "width:", ",").c_str());

  return 0;
}

template <typename Dtype>
std::string InputOp<Dtype>::GenOpParamStr() const {
  std::stringstream param_str;
  param_str << "num:" << param_.num << ",channels:" << param_.channels << ",height:" << param_.height << ",width:" << param_.width << ",";
  return param_str.str();
}

template <typename Dtype>
int InputOp<Dtype>::InferOutShape(std::vector<int> &in_shape, std::vector<int> &out_shape) {
  // if input shape has not been setted
  if (in_shape.size() != 4) {
    in_shape.clear();
    in_shape.push_back(param_.num);
    in_shape.push_back(param_.channels);
    in_shape.push_back(param_.height);
    in_shape.push_back(param_.width);
  }

  else if (in_shape[1] != param_.channels ||
    in_shape[2] != param_.height ||
    in_shape[3] != param_.width) {
    DLOG_ERR("[ InputOp::InferOutShape ]: in_shape[1] != param_.channels_ || \
             				in_shape[2] != param_.height_ || in_shape[3] != param_.width_ \n");

    return -1;
  }
  else
    param_.num = in_shape[0];

  out_shape = in_shape;
  return 0;
}

template <typename Dtype>
int InputOp<Dtype>::AllocBuf4Node(const std::vector<int> &in_shape,
  const std::vector<int> &out_shape,
  std::vector<std::shared_ptr<Tensor<Dtype>>> &data) const {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ InputOp::AllocBuf4Node ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  data.clear();
  data.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  return 0;
}

template <typename Dtype>
int InputOp<Dtype>::AllocOpBuf4Train(const std::vector<int> &in_shape, const std::vector<int> &out_shape) {
  if (in_shape[tind::eNum] <= 0 || in_shape[tind::eChannels] <= 0 ||
    in_shape[tind::eHeight] <= 0 || in_shape[tind::eWidth] <= 0 ||
    in_shape[tind::eNum] > 5000 || in_shape[tind::eChannels] > 5000 ||
    in_shape[tind::eHeight] > 5000 || in_shape[tind::eWidth] > 5000) {
    DLOG_ERR("[ InputOp::AllocOpBuf4Train ]: in_shape is invalid -> (%d, %d, %d, %d) \n",
      in_shape[tind::eNum], in_shape[tind::eChannels], in_shape[tind::eHeight], in_shape[tind::eWidth]);
    return -1;
  }

  diff_.clear();
  diff_.push_back(std::make_shared<Tensor<Dtype>>(in_shape));

  return 0;
}

template <typename Dtype>
void InputOp<Dtype>::Forward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  for (int i = 0; i < prev.size(); i++)
    prev[i]->CopyDataTo(*next[i], tind::eHost2Host);
}

template <typename Dtype>
void InputOp<Dtype>::Backward(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {
  //the Backward operation in InputOp is empty
}

#ifdef USE_CUDA
template <typename Dtype>
void InputOp<Dtype>::Forward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next) {
  for (int i = 0; i < prev.size(); i++)
    prev[i]->CopyDataTo(*next[i], tind::eDevice2Device);
}

template <typename Dtype>
void InputOp<Dtype>::Backward_gpu(
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &prev_diff,
  const std::vector<std::shared_ptr<Tensor<Dtype>>> &next_diff) {
  //the Backward operation in InputOp is empty
}
#endif
INSTANTIATE_CLASS(InputOp);
}//namespace
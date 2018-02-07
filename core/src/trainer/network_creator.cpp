////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Create network
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "trainer/network_creator.h"

namespace dlex_cnn {
//////////////// Input //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateInputNode(std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> input_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Input");
  if (input_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::InputOp<Dtype> *>(input_s.get())->SetOpParam(param);

  std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > input;
  input.push_back(input_s);

  network.AddNode(name, input);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateInputNode(std::string name, InputOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> input_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Input");
  if (input_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::InputOp<Dtype> *>(input_s.get())->SetOpParam(param);

  std::vector < std::shared_ptr<dlex_cnn::Op<Dtype>> > input;
  input.push_back(input_s);

  network.AddNode(name, input);
  return 0;
}

//////////////// Inner Product //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateInnerProductNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> fc_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("InnerProduct");
  if (fc_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::InnerProductOp<Dtype> *>(fc_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> fc;
  fc.push_back(fc_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, fc, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateInnerProductNode(std::string in_node, std::string name, InnerProductOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> fc_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("InnerProduct");
  if (fc_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::InnerProductOp<Dtype> *>(fc_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> fc;
  fc.push_back(fc_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, fc, inNodeNames);
  return 0;
}

//////////////// Convolution //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateConvNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> conv_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Convolution");
  if (conv_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::ConvolutionOp<Dtype> *>(conv_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> conv;
  conv.push_back(conv_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, conv, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateConvNode(std::string in_node, std::string name, ConvolutionOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> conv_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Convolution");
  if (conv_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::ConvolutionOp<Dtype> *>(conv_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> conv;
  conv.push_back(conv_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, conv, inNodeNames);
  return 0;
}

//////////////// Deconvolution //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateDeconvNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> deconv_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Deconvolution");
  if (deconv_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::DeconvolutionOp<Dtype> *>(deconv_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> deconv;
  deconv.push_back(deconv_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, deconv, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateDeconvNode(std::string in_node, std::string name, DeconvolutionOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> deconv_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Deconvolution");
  if (deconv_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::DeconvolutionOp<Dtype> *>(deconv_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> deconv;
  deconv.push_back(deconv_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, deconv, inNodeNames);
  return 0;
}

//////////////// Activation //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateActivationNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> act_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Activation");
  if (act_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::ActivationOp<Dtype> *>(act_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> act;
  act.push_back(act_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, act, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateActivationNode(std::string in_node, std::string name, ActivationOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> act_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Activation");
  if (act_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::ActivationOp<Dtype> *>(act_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> act;
  act.push_back(act_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, act, inNodeNames);
  return 0;
}

//////////////// Pooling //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreatePoolNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> pool_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Pooling");
  if (pool_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::PoolingOp<Dtype> *>(pool_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> pool;
  pool.push_back(pool_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, pool, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreatePoolNode(std::string in_node, std::string name, PoolingOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> pool_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Pooling");
  if (pool_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::PoolingOp<Dtype> *>(pool_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> pool;
  pool.push_back(pool_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, pool, inNodeNames);
  return 0;
}

//////////////// Softmax Cross Entropy Loss //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateSoftmaxLossNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  //std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("SoftmaxCrossEntropyLossH");
  //if (sm_s == NULL)
  //	return 1;

  //dynamic_cast<dlex_cnn::SoftmaxCrossEntropyLossHOp<Dtype> *>(sm_s.get())->SetOpParam(param);

  //std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> sm;
  //sm.push_back(sm_s);

  //std::vector<std::string> inNodeNames;
  //inNodeNames.push_back(in_node);

  //network.AddNode(name, sm, inNodeNames);

  //////////// way 2 /////////////

  std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Softmax");
  if (sm_s == NULL)
    return 1;
  dlex_cnn::SoftmaxOpParam softmaxParam;
  dynamic_cast<dlex_cnn::SoftmaxOp<Dtype> *>(sm_s.get())->SetOpParam(softmaxParam);

  std::shared_ptr<dlex_cnn::Op<Dtype>> cel_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("CrossEntropyLoss");
  if (cel_s == NULL)
    return 1;
  dlex_cnn::CrossEntropyLossOpParam CELParam;
  dynamic_cast<dlex_cnn::CrossEntropyLossOp<Dtype> *>(cel_s.get())->SetOpParam(CELParam);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> sm;
  sm.push_back(sm_s);
  sm.push_back(cel_s);
  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, sm, inNodeNames);

  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateSoftmaxLossNode(std::string in_node, std::string name, SoftmaxCrossEntropyLossHOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> sm_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("SoftmaxCrossEntropyLossH");
  if (sm_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::SoftmaxCrossEntropyLossHOp<Dtype> *>(sm_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> sm;
  sm.push_back(sm_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, sm, inNodeNames);
  return 0;
}

//////////////// Output //////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateOutputNode(std::string in_node, std::string name, std::string param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> out_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Output");
  if (out_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::OutputOp<Dtype> *>(out_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> out;
  out.push_back(out_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, out, inNodeNames);
  return 0;
}

template <typename Dtype>
int NetCreator<Dtype>::CreateOutputNode(std::string in_node, std::string name, OutputOpParam param, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Op<Dtype>> out_s = dlex_cnn::OpFactory<Dtype>::GetInstance().CreateOpByType("Output");
  if (out_s == NULL)
    return 1;

  dynamic_cast<dlex_cnn::OutputOp<Dtype> *>(out_s.get())->SetOpParam(param);

  std::vector<std::shared_ptr<dlex_cnn::Op<Dtype>>> out;
  out.push_back(out_s);

  std::vector<std::string> inNodeNames;
  inNodeNames.push_back(in_node);

  network.AddNode(name, out, inNodeNames);
  return 0;
}

/////////////// Optimizer /////////////////
template <typename Dtype>
int NetCreator<Dtype>::CreateOptimizer(std::string opt_type, NetWork<Dtype> &network) {
  std::shared_ptr<dlex_cnn::Optimizer<Dtype>> optimizer;
  dlex_cnn::Optimizer<Dtype>::getOptimizerByStr(opt_type, optimizer);
  network.SetOptimizer(optimizer);
  return 0;
}

INSTANTIATE_CLASS(NetCreator);
}
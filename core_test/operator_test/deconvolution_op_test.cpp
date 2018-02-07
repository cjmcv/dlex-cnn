////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test Deconvolution operator.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_cnn.h"
#include "common/tools.h"
#include "deconvolution_op_test.h"

//#include "../core/operator/convolution_op.h"

#ifdef USE_OP_TEST
namespace dlex_cnn {

template <typename Dtype>
void DeconvolutionOpTest<Dtype>::Exec() {
  bool isTestGpu = false;
  RegisterOpClass();

  std::shared_ptr<dlex_cnn::Op<float>> conv1_s = dlex_cnn::OpFactory<float>::GetInstance().CreateOpByType("Deconvolution");
  dlex_cnn::DeconvolutionOpParam deconv_param;
  deconv_param.blas_enable = true;
  deconv_param.kernel_channels = 3;
  deconv_param.kernel_h = 2;
  deconv_param.kernel_w = 2;
  deconv_param.pad_h = 0;
  deconv_param.pad_w = 0;
  deconv_param.stride_h = 1;
  deconv_param.stride_w = 1;
  deconv_param.dilation_h = 1;
  deconv_param.dilation_w = 1;

  dlex_cnn::DeconvolutionOp<float>* conv1 = dynamic_cast<dlex_cnn::DeconvolutionOp<float> *>(conv1_s.get());
  conv1->SetOpParam(deconv_param);

  int is[4] = { 1, 2, 3, 3 };	//3
  std::vector<int> in_shape;
  for (int i = 0; i < 4; i++)
    in_shape.push_back(is[i]);

  std::vector<int> out_shape;
  conv1->InferOutShape(in_shape, out_shape);

  std::vector<std::shared_ptr<Tensor<float>>> in_data_vec;
  conv1->AllocBuf4Node(in_shape, out_shape, in_data_vec);

  normal_distribution_init<float>(in_data_vec[1]->get_size()[tind::e4D], 0.0f, 0.1f, (float *)in_data_vec[1]->GetPushCpuData());
  if (conv1->param_.blas_enable)
    set_cpu<float>(in_data_vec[2]->get_size()[tind::e4D], 0.0f, (float *)in_data_vec[2]->GetPushCpuData());

  // input (ic2, ih3, iw3)
  float in_buffer[] = { 1, 2, 0, 1, 1, 3, 0, 2, 2, 0, 2, 1, 0, 3, 2, 1, 1, 0 }; //
  //float in_buffer[] = {1,0,0,0, 0,1,0,0, 0,0,3,0, 0,0,0,1};
  float *in_data = (float *)in_data_vec[0]->GetPushCpuData();
  for (int i = 0; i < 1 * 2 * 3 * 3; i++)
    in_data[i] = in_buffer[i];
  // weight (kn2 = ic2, kc3, kh2, kw2) 
  float w_buffer[] = { 1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 3, 1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0 };	//2*3*2*2
  //float w_buffer[] = {4,5,3,4};	//2*2*2*2
  float *w_data = (float *)in_data_vec[1]->GetPushCpuData();
  for (int i = 0; i < 2 * 3 * 2 * 2; i++)
    w_data[i] = w_buffer[i];
  // bias ()
  //float b_buffer[] = { 1, 2, 0, 1, 1, 3, 0, 2, 2, 0, 2, 1, 0, 3, 2, 1, 1, 0, 1, 2, 1, 0, 1, 3, 3, 3, 2 };
  //float *in_data = (float *)in_data_vec[0]->GetPushCpuData();
  //for (int i = 0; i < in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3]; i++)
  //	in_data[i] = in_buffer[i];

  conv1->AllocOpBuf4Train(in_shape, out_shape);

  // Test Forward.
  std::vector<std::shared_ptr<Tensor<float>>> out_data_vec;
  out_data_vec.push_back(std::make_shared<Tensor<float>>(out_shape));

  if (isTestGpu) {
#ifdef USE_CUDA
    conv1->Forward_gpu(in_data_vec, out_data_vec);
#else
    DLOG_ERR("The marco USE_CUDA is closed, please open it for testing in GPU.");
#endif
  }
  else
    conv1->Forward(in_data_vec, out_data_vec);

  MatrixShow_float("A", (float *)in_data_vec[0]->GetPushCpuData(),
    in_data_vec[0]->get_shape()[tind::eNum],
    in_data_vec[0]->get_shape()[tind::eChannels],
    in_data_vec[0]->get_shape()[tind::eHeight],
    in_data_vec[0]->get_shape()[tind::eWidth]);
  MatrixShow_float("W", (float *)in_data_vec[1]->GetPushCpuData(),
    in_data_vec[1]->get_shape()[tind::eNum],
    in_data_vec[1]->get_shape()[tind::eChannels],
    in_data_vec[1]->get_shape()[tind::eHeight],
    in_data_vec[1]->get_shape()[tind::eWidth]);
  MatrixShow_float("B", (float *)out_data_vec[0]->GetPushCpuData(),
    out_data_vec[0]->get_shape()[tind::eNum],
    out_data_vec[0]->get_shape()[tind::eChannels],
    out_data_vec[0]->get_shape()[tind::eHeight],
    out_data_vec[0]->get_shape()[tind::eWidth]);

  // Test Backward.
  std::vector<std::shared_ptr<Tensor<Dtype>>> in_diff_vec;
  in_diff_vec.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
  std::vector<std::shared_ptr<Tensor<Dtype>>> out_diff;
  out_diff.push_back(std::make_shared<Tensor<Dtype>>(out_shape));

  if (isTestGpu) {
#ifdef USE_CUDA
    conv1->Backward_gpu(in_data_vec, out_data_vec, in_diff_vec, out_data_vec);
#else
    DLOG_ERR("The marco USE_CUDA is closed, please open it for testing in GPU.");
#endif
  }
  else
    conv1->Backward(in_data_vec, out_data_vec, in_diff_vec, out_data_vec);

  MatrixShow_float("C", (float *)in_diff_vec[0]->GetPushCpuData(),
    in_shape[tind::eNum], in_shape[tind::eChannels],
    in_shape[tind::eHeight], in_shape[tind::eWidth]);

  const std::vector<int> kernel_shape = in_data_vec[1]->get_shape();
  MatrixShow_float("weight gradient", (float *)conv1->gradient_[0]->GetPushCpuData(),
    kernel_shape[tind::eNum], kernel_shape[tind::eChannels],
    kernel_shape[tind::eHeight], kernel_shape[tind::eWidth]);

  const std::vector<int> bias_shape = in_data_vec[2]->get_shape();
  MatrixShow_float("bias gradient", (float *)conv1->gradient_[1]->GetPushCpuData(),
    bias_shape[tind::eNum], bias_shape[tind::eChannels],
    bias_shape[tind::eHeight], bias_shape[tind::eWidth]);
}

}

void TestDeconv() {
  dlex_cnn::DeconvolutionOpTest<float> deconv_test;
  deconv_test.Exec();
  system("pause");
}
#endif

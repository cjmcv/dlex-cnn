////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test Pooling operator.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "dlex_cnn.h"
#include "common/tools.h"
#include "pooling_op_test.h"

#ifdef USE_OP_TEST
namespace dlex_cnn {

template <typename Dtype>
void PoolingOpTest<Dtype>::Exec() {
  bool isTestGpu = true;
  RegisterOpClass();

  std::shared_ptr<dlex_cnn::Op<float>> pool_s = dlex_cnn::OpFactory<float>::GetInstance().CreateOpByType("Pooling");
  dlex_cnn::PoolingOpParam pooling_param;
  pooling_param.global_pooling = false;
  pooling_param.kernel_h = 3;
  pooling_param.kernel_w = 3;
  pooling_param.pad_h = 0;
  pooling_param.pad_w = 0;
  pooling_param.stride_h = 3;
  pooling_param.stride_w = 3;
  pooling_param.pooling_type = dlex_cnn::tind::eMAX;
  dlex_cnn::PoolingOp<float>* pool = dynamic_cast<dlex_cnn::PoolingOp<float> *>(pool_s.get());
  pool->SetOpParam(pooling_param);

  int is[4] = { 1, 1, 10, 10 };
  std::vector<int> in_shape;
  for (int i = 0; i < 4; i++)
    in_shape.push_back(is[i]);

  std::vector<int> out_shape;
  pool->InferOutShape(in_shape, out_shape);

  std::vector<std::shared_ptr<Tensor<float>>> in_data_vec;
  pool->AllocBuf4Node(in_shape, out_shape, in_data_vec);
  pool->AllocOpBuf4Train(in_shape, out_shape);

  // input (ic3, ih3, iw3)
  float *in_data = (float *)in_data_vec[0]->GetCpuData();
  for (int i = 0; i < 1 * 1 * 10 * 10; i++)
    in_data[i] = i;

  // Test Forward.
  std::vector<std::shared_ptr<Tensor<float>>> out_data_vec;
  out_data_vec.push_back(std::make_shared<Tensor<float>>(out_shape));

  if (isTestGpu) {
#ifdef USE_CUDA
    pool->Forward_gpu(in_data_vec, out_data_vec);
#else
    DLOG_ERR("The marco USE_CUDA is closed, please open it for testing in GPU.");
#endif
  }
  else
    pool->Forward(in_data_vec, out_data_vec);

  MatrixShow_float("A", (float *)in_data_vec[0]->GetPushCpuData(),
    in_data_vec[0]->get_shape()[tind::eNum],
    in_data_vec[0]->get_shape()[tind::eChannels],
    in_data_vec[0]->get_shape()[tind::eHeight],
    in_data_vec[0]->get_shape()[tind::eWidth]);
  MatrixShow_float("B", (float *)out_data_vec[0]->GetPushCpuData(),
    out_data_vec[0]->get_shape()[tind::eNum],
    out_data_vec[0]->get_shape()[tind::eChannels],
    out_data_vec[0]->get_shape()[tind::eHeight],
    out_data_vec[0]->get_shape()[tind::eWidth]);
  if (pooling_param.pooling_type = dlex_cnn::tind::eMAX)
    MatrixShow_int("C", (int *)pool->max_idx_map_->GetPushCpuData(),
    out_data_vec[0]->get_shape()[tind::eNum],
    out_data_vec[0]->get_shape()[tind::eChannels],
    out_data_vec[0]->get_shape()[tind::eHeight],
    out_data_vec[0]->get_shape()[tind::eWidth]);

  // Test Backward.
  std::vector<std::shared_ptr<Tensor<Dtype>>> in_diff_vec;
  in_diff_vec.push_back(std::make_shared<Tensor<Dtype>>(in_shape));
  std::vector<std::shared_ptr<Tensor<Dtype>>> out_diff_vec;
  out_diff_vec.push_back(std::make_shared<Tensor<Dtype>>(out_shape));

  if (isTestGpu) {
#ifdef USE_CUDA
    pool->Backward_gpu(in_data_vec, out_data_vec, in_diff_vec, out_data_vec);
#else
    DLOG_ERR("The marco USE_CUDA is closed, please open it for testing in GPU.");
#endif
  }
  else
    pool->Backward(in_data_vec, out_data_vec, in_diff_vec, out_data_vec);
  MatrixShow_float("D", (float *)in_diff_vec[0]->GetPushCpuData(),
    in_shape[tind::eNum],
    in_shape[tind::eChannels],
    in_shape[tind::eHeight],
    in_shape[tind::eWidth]);
}

}

void TestPool() {
  dlex_cnn::PoolingOpTest<float> pool_test;
  pool_test.Exec();
}
#endif

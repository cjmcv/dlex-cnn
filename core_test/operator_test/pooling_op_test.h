////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test Pooling operator.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_POOLING_HPP_
#define DLEX_TEST_POOLING_HPP_

#ifdef USE_OP_TEST

#include "operator/pooling_op.h"
namespace dlex_cnn {
template <typename Dtype>
class PoolingOpTest {
  template<typename T> friend class PoolingOp;

public:
  PoolingOpTest() {};
  virtual ~PoolingOpTest() {};

public:
  void Exec();
};
}
void TestPool();

#endif // USE_OP_TEST

#endif


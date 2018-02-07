////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Test Deconvolution operator.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TEST_DECONV_HPP_
#define DLEX_TEST_DECONV_HPP_

#ifdef USE_OP_TEST

#include "operator/deconvolution_op.h"
namespace dlex_cnn {
template <typename Dtype>
class DeconvolutionOpTest {
  template<typename T> friend class DeconvolutionOp;

public:
  DeconvolutionOpTest() {};
  virtual ~DeconvolutionOpTest() {};

public:
  void Exec();
};
}
void TestDeconv();

#endif //USE_OP_TEST

#endif

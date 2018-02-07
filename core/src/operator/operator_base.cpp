////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "operator/operator_base.h"

namespace dlex_cnn {
template <typename Dtype>
Op<Dtype>::Op() {}

template <typename Dtype>
Op<Dtype>::~Op() {}

INSTANTIATE_CLASS(Op);
}
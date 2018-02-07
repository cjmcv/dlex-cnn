////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Common.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "common.h"

namespace dlex_cnn {
std::string FetchSubStr(std::string &src_str, std::string start_str, std::string end_str) {
  int start_idx = src_str.find(start_str) + start_str.length();
  int end_idx = src_str.find(end_str, start_idx);
  return src_str.substr(start_idx, end_idx - start_idx);
}
}

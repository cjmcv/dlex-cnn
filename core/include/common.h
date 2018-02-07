////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Common.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_COMMON_HPP_
#define DLEX_COMMON_HPP_

#include <iostream>

namespace dlex_cnn {
#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38F 
#endif

// Class instantiate
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

#define INSTANTIATE_CLASS_NOR(classname) \
  char gInstantiationGuard##classname; \
  template class classname<int>; \
  template class classname<float>; \
  template class classname<double>

// Log control
#define DLOG_ERR(format, ...) fprintf(stderr,"ERROR: "#format"\n", ##__VA_ARGS__);
#define DLOG_INFO(format, ...) fprintf(stdout,"INFO: "#format"\n", ##__VA_ARGS__);
#define DLOG_WARN(format, ...) fprintf(stdout,"WARN: "#format"\n", ##__VA_ARGS__);

// Variable inspection
#define DCHECK(val) (((val)==0)? false:true)

#define DCHECK_EQ(val1, val2) (((val1)==(val2))? true:false)
#define DCHECK_NE(val1, val2) (((val1)!=(val2))? true:false)

#define DCHECK_LE(val1, val2) (((val1)<=(val2))? true:false)
#define DCHECK_LT(val1, val2) (((val1)< (val2))? true:false)

#define DCHECK_GE(val1, val2) (((val1)>=(val2))? true:false)
#define DCHECK_GT(val1, val2) (((val1)> (val2))? true:false)

// Error Code
enum EN_ErrCode {
	CUDA_DCHECK_EXC = 1,
	CURAND_DCHECK_EXC = 2,
	CUBLAS_DCHECK_EXC = 3
};

// String process
std::string FetchSubStr(std::string &src_str, std::string start_str, std::string end_str);

}

#endif //DLEX_COMMON_HPP_

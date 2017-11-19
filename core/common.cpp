////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "common.h"

namespace dlex_cnn
{
	std::string fetchSubStr(std::string &srcStr, std::string startStr, std::string endStr)
	{
		int start_idx = srcStr.find(startStr) + startStr.length();
		int end_idx = srcStr.find(endStr, start_idx);
		return srcStr.substr(start_idx, end_idx - start_idx);
	}
}
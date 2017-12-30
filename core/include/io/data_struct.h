////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_DATA_STRUCT_HPP_
#define DLEX_DATA_STRUCT_HPP_

namespace dlex_cnn
{
	typedef struct IMAGE_DATUM
	{
		unsigned char *pdata;
		int channels;
		int height;
		int width;
	}IMAGE_DATUM;
}
#endif //DLEX_PREFETCH_HPP_
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_NET_TYPICAL_HPP_
#define DLEX_NET_TYPICAL_HPP_
//
#include <iostream>
#include "network_creator.h"

namespace dlex_cnn
{
	class TypicalNet
	{
    public:
		template <typename Dtype>
		int mlp(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network);
		template <typename Dtype>
		int lenet(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network);
		template <typename Dtype>
		int mix(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network);
	};
}

#endif

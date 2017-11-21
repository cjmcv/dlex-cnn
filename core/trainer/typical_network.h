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
	template <typename Dtype>
	class TypicalNet
	{
	public:
		TypicalNet() {};
		virtual ~TypicalNet() {};

	public:
		int mlp(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network);
		int lenet(const int num, const int channels, const int height, const int width, NetWork<Dtype> &network);

	};

	template int TypicalNet<float>::mlp(const int num, const int channels, const int height, const int width, NetWork<float> &network);
	template int TypicalNet<double>::mlp(const int num, const int channels, const int height, const int width, NetWork<double> &network);
}

#endif
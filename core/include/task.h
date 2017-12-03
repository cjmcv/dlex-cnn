////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Mainly for providing the device mode.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TASK_HPP_
#define DLEX_TASK_HPP_

#include <iostream>

namespace dlex_cnn
{
	namespace tind
	{
		enum Mode { CPU, GPU };
	}

	class Task
	{
	public:
		Task();
		virtual ~Task();

	public:
		static Task& Get();
		inline static void set_mode(tind::Mode mode) { Get().device_mode_ = mode; }
		inline static tind::Mode mode() { return Get().device_mode_; }

	private:
		tind::Mode device_mode_;
	};
}

#endif
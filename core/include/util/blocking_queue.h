////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_BLOCKING_QUEUE_HPP_
#define DLEX_BLOCKING_QUEUE_HPP_

#include "common.h"
//#include <thread>
#include <queue>
#include <condition_variable>

namespace dlex_cnn
{
	template <typename T>
	class BlockingQueue {
	public:
		BlockingQueue();
		virtual ~BlockingQueue();

		void push(const T& t);
		bool try_pop(T* t);
		void wait_and_pop(T* t);

		bool empty() const;

	private:
		mutable std::mutex mutex_;
		std::condition_variable cond_var_;
		std::queue<T> queue_;
	};
}
#endif //DLEX_BLOCKING_QUEUE_HPP_
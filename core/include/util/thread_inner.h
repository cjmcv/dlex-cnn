////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_THREAD_INNER_HPP_
#define DLEX_THREAD_INNER_HPP_

#include "common.h"
#include <thread>
#include <memory>


namespace dlex_cnn
{
	class ThreadInner {
	public:
		ThreadInner();
		virtual ~ThreadInner();

		void startInnerThread();
		void stopInnerThread();

		bool is_started() const;

	protected:
		virtual void entryInnerThread() {}
		bool must_stop();

	private:
		std::shared_ptr<std::thread> thread_;
	};
}
#endif //DLEX_THREAD_INNER_HPP_
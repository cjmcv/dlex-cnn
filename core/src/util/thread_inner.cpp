////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Internal thread. It is mainly used in prefetcher.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/thread_inner.h"

namespace dlex_cnn
{

	ThreadInner::ThreadInner()
		: thread_(), interrupt_flag_(false)
	{

	}
	ThreadInner::~ThreadInner()
	{
		stopInnerThread();
	}

	inline bool ThreadInner::is_started() const 
	{
		return thread_ && thread_->joinable();
	}

	bool ThreadInner::must_stop() 
	{
		if (thread_ && interrupt_flag_)// && thread_->interruption_requested();
		{
			interrupt_flag_ = false;
			return true;
		}
		else
			return false;
	}

	void ThreadInner::startInnerThread()
	{
		if (!DCHECK(!is_started()))
			DLOG_ERR("Threads should persist and not be restarted.");
		try {
			thread_.reset(new std::thread(&ThreadInner::entryInnerThread, this));
		}
		catch (std::exception& e) {
			DLOG_ERR("Thread exception: %s", e.what());
		}
	}

	void ThreadInner::stopInnerThread()
	{
		if (is_started()) 
		{
			//thread_->interrupt();
			interrupt_flag_ = true;
			try 
			{
				thread_->join();
			}
			catch (std::exception& e) 
			{
				DLOG_ERR("Thread exception: %s", e.what());
			}
		}
	}

}
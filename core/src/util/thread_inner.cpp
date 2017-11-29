////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "util/thread_inner.h"

namespace dlex_cnn
{

	ThreadInner::ThreadInner()
		: thread_()
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

	//bool ThreadInner::must_stop() {
	//	return thread_ && thread_->interruption_requested();
	//}

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
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "thread_pool.h"

namespace dlex_cnn
{
	// the constructor just launches some amount of workers_
	// m_exec.push_back(std::thread(std::mem_fn(&GxxMtcnn::InnerDetect), this));
	ThreadPool::ThreadPool()
		: is_created_(false)
	{
	}
	// the destructor joins all threads
	ThreadPool::~ThreadPool()
	{
		if (is_created_ == true)
			clearPool();
	}

	void ThreadPool::createThreads(int thread_num)
	{
		is_stop_ = false;
		if (is_created_ == true)
		{
			if (workers_.size() == thread_num)
				return;
			else
			{
				clearPool();
				is_created_ = false;
			}
		}
		//printf("2\n");
		for (int i = 0; i < thread_num; ++i)
			workers_.emplace_back(
			[this]
		{
			//printf("aa\n");
			for (;;)
			{
				std::function<void()> task;

				{
					std::unique_lock<std::mutex> lock(this->queue_mutex_);
					this->condition_.wait(lock,
						[this]{ return this->is_stop_ || !this->tasks_.empty(); });
					if (this->is_stop_ && this->tasks_.empty())
						return;
					task = std::move(this->tasks_.front());
					this->tasks_.pop();
				}

				task();
			}
		}
		);
		//printf("3\n");
		is_created_ = true;
	}

	void ThreadPool::clearPool()
	{
		{
			std::unique_lock<std::mutex> lock(queue_mutex_);
			is_stop_ = true;
		}
		condition_.notify_all();
		for (std::thread &worker : workers_)
			worker.join();

		workers_.clear();
		tasks_ = decltype(tasks_)();

		is_created_ = false;
	}

	void ThreadPool::exec(std::function<void(const int, const int)> func, const int number)
	{
		if (number <= 0)
		{
			DLOG_ERR("[ ThreadPool::exec ]: number <= 0\n");
			return;
		}
		const int threads_num = workers_.size();
		if (threads_num <= 1 || number <= 1)
		{
			func(0, number);
		}
		else
		{
			const int datum_per_thread = number / threads_num;
			const int datum_remainder = number - datum_per_thread * threads_num;

			int start_num_idx = 0;
			std::vector<std::future<void>> futures;
			for (int i = 0; i < threads_num; i++)
			{
				int stop_num_idx = start_num_idx + datum_per_thread;
				if (i < datum_remainder)
					stop_num_idx = stop_num_idx + 1;

				futures.emplace_back(enqueue(func, start_num_idx, stop_num_idx));

				start_num_idx = stop_num_idx;
				if (stop_num_idx >= number)
					break;
			}

			for (int i = 0; i < futures.size(); i++)
				futures[i].wait();
		}
	}

}
////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Used for multi-threading acceleration.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_THREAD_POOL_HPP_
#define DLEX_THREAD_POOL_HPP_

#include "common.h"

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

namespace dlex_cnn
{
	class ThreadPool {
	public:
		ThreadPool();
		~ThreadPool();
		void createThreads(int thread_num);
		void clearPool();
		void exec(std::function<void(const int, const int)> func, const int number);

		template<class F, class... Args>
		auto enqueue(F&& f, Args&&... args)
			->std::future<typename std::result_of<F(Args...)>::type>;

	private:
		// need to keep track of threads so we can join them
		std::vector< std::thread > workers_;
		// the task queue
		std::queue< std::function<void()> > tasks_;

		// synchronization
		std::mutex queue_mutex_;
		std::condition_variable condition_;
		bool is_stop_;
		bool is_created_;
	};

	// add new work item to the pool
	template<class F, class... Args>
	auto ThreadPool::enqueue(F&& f, Args&&... args)
		-> std::future<typename std::result_of<F(Args...)>::type>
	{
		if (is_created_ == false)
			createThreads(2);	// creating 2 threads by default

		using return_type = typename std::result_of<F(Args...)>::type;

		auto task = std::make_shared< std::packaged_task<return_type()> >(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
			);

		std::future<return_type> res = task->get_future();
		{
			std::unique_lock<std::mutex> lock(queue_mutex_);

			// don't allow enqueueing after stopping the pool
			if (is_stop_)
				throw std::runtime_error("enqueue on stopped ThreadPool");

			tasks_.emplace([task](){ (*task)(); });
		}
		condition_.notify_one();
		return res;
	}

}
#endif //DLEX_THREAD_POOL_HPP_
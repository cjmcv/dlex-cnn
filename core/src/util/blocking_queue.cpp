////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Designed for thread safety.
//          It works mainly in prefetcher.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#include "tensor.h"
#include "util/blocking_queue.h"

namespace dlex_cnn {
template <typename T>
BlockingQueue<T>::BlockingQueue() {}

template <typename T>
BlockingQueue<T>::~BlockingQueue() {}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  std::unique_lock <std::mutex> lock(mutex_);
  queue_.push(t);
  lock.unlock();
  cond_var_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::empty() const {
  std::unique_lock <std::mutex> lock(mutex_);
  return queue_.empty();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  if (queue_.empty())
    return false;

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
void BlockingQueue<T>::wait_and_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  while (queue_.empty())
    cond_var_.wait(lock);

  *t = queue_.front();
  queue_.pop();
}

template class BlockingQueue<std::pair < std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>> >*>;
template class BlockingQueue<std::pair < std::shared_ptr<Tensor<double>>, std::shared_ptr<Tensor<double>> >*>;
template class BlockingQueue< std::shared_ptr< Tensor<float> > >;
template class BlockingQueue< std::shared_ptr< Tensor<double> > >;
INSTANTIATE_CLASS(BlockingQueue);
}
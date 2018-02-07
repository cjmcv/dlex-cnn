////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Internal thread. It is mainly used in prefetcher.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_THREAD_INNER_HPP_
#define DLEX_THREAD_INNER_HPP_

#include "common.h"
#include <thread>
#include <memory>


namespace dlex_cnn {
class ThreadInner {
public:
  ThreadInner();
  virtual ~ThreadInner();

  void StartInnerThread();
  void StopInnerThread();

  // To chech wether the inner thread has been started. 
  bool is_started() const;

protected:
  // Virtual function, should be override by the classes which needs a internal thread to assist.
  virtual void EntryInnerThread() {}
  bool must_stop();

private:
  bool interrupt_flag_;
  std::shared_ptr<std::thread> thread_;
};
}
#endif //DLEX_THREAD_INNER_HPP_
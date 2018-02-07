////////////////////////////////////////////////////////////////
// > Copyright (c) 2017 by Contributors. 
// > https://github.com/cjmcv
// > brief  Timer.
// > author Jianming Chen
////////////////////////////////////////////////////////////////

#ifndef DLEX_TIMER_HPP_
#define DLEX_TIMER_HPP_

#include <chrono>

namespace dlex_cnn {
class Timer {
public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;
  Timer() { Start(); }

  // Starts a timer.
  inline void Start() { start_time_ = clock::now(); }
  inline float NanoSeconds() {
    return std::chrono::duration_cast<ns>(clock::now() - start_time_).count();
  }

  // Returns the elapsed time in milliseconds.
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }

  //brief Returns the elapsed time in microseconds.
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }

  //Returns the elapsed time in seconds.
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
  std::chrono::time_point<clock> start_time_;
};
}
#endif //DLEX_TIMER_HPP_
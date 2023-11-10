#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <utility>
#include <vector>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime_api.h"

extern bool performance_test;

namespace util {

#define ASSERT(x)                                                     \
  do {                                                                \
    if (!(x)) {                                                       \
      std::fprintf(stderr, "%s:%d %s: CModelAssertion `%s' failed\n", \
                   __FILE__, __LINE__, __func__, #x);                 \
      abort();                                                        \
    }                                                                 \
  } while (0)

#define call(call_func)                                                          \
    do {                                                                           \
        if (true) {                                                      \
            int warm_times = 30, exec_times = 10;                                      \
            printf("=========================================================\n");     \
            printf("Warming up[%d times] ...", warm_times);                            \
                                                                                                                                                                 \
            for (int i=0; i<warm_times; i++) {                                         \
                call_func;                                                               \
                CUDA_CHECK(cudaDeviceSynchronize());                                     \
                CUDA_CHECK(cudaGetLastError());                                        \
            }                                                                          \
            printf("Done\n");                                                          \
                                                                                                                                                                 \
            Timer timer;                                                               \
            for (int i=0; i<exec_times; i++) {                                         \
                timer.start();                                                           \
                call_func;                                                               \
                CUDA_CHECK(cudaDeviceSynchronize());                                     \
                CUDA_CHECK(cudaGetLastError());                                        \
                timer.stop();                                                            \
                timer.ShowLastTime("");                          \
            }                                                                          \
            timer.ShowAverageBatchTime(exec_times, call_times);                         \
            timer.ShowAverageTime(exec_times);                                          \
      } else {                                                                     \
        call_func;                                                                 \
      }                                                                            \
    } while (0);

void cuda_check(cudaError_t state, std::string file, int line);
#define CUDA_CHECK(x) cuda_check(x, __FILE__, __LINE__)

class Timer {
 private:
  std::chrono::nanoseconds GetTimerValue() const {
    if (active_) {
      return (clock_.now() - start_time_);
    } else {
      return last_time_;
    }
  }

  std::chrono::high_resolution_clock clock_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::nanoseconds total_time_ =
      std::chrono::high_resolution_clock::duration::zero();
  std::chrono::nanoseconds last_time_ =
      std::chrono::high_resolution_clock::duration::zero();
  uint32_t count_ = 0;
  bool active_ = false;
  bool time_verbose_{true};
  size_t min_us_ = 0;
  size_t max_us_ = 0;

 public:
  Timer(bool verbose = true) : time_verbose_(verbose) {}
  void start() {
    if (active_ == false) {
      active_ = true;
      count_++;
      start_time_ = clock_.now();
    }
  }

  void stop() {
    if (active_) {
      auto end_time = clock_.now();
      last_time_ = end_time - start_time_;
      total_time_ += last_time_;
      active_ = false;

      size_t last_us = GetLastMicroseconds();
      if (count_ == 1) {
      	min_us_ = max_us_ = last_us;
      } else {
      	min_us_ = min_us_ < last_us ? min_us_ : last_us;
      	max_us_ = max_us_ > last_us ? max_us_ : last_us;
      }
    }
  }

  bool IsVerbose() const { return time_verbose_; }

  uint32_t GetCallCount() const { return count_; }

  size_t GetSeconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(GetTimerValue())
        .count();
  }
  size_t GetMilliSeconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               GetTimerValue())
        .count();
  }
  size_t GetMicroSeconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               GetTimerValue())
        .count();
  }
  size_t GetNanoSeconds() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(GetTimerValue())
        .count();
  }

  size_t GetTotalSeconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(total_time_)
        .count();
  }
  size_t GetTotalMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(total_time_)
        .count();
  }

  size_t GetTotalMicroseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(total_time_)
        .count();
  }

  size_t GetLastMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(last_time_)
        .count();
  }

  size_t GetLastMicroseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(last_time_)
        .count();
  }

  void ShowLastTime(std::string title) {
		size_t time = GetLastMicroseconds();
		std::cout << title << "time cost: " << float(time) / 1000 << +"ms" << std::endl;
  }

  void ShowAverageTime(int exec_times) {
        size_t time = GetTotalMicroseconds();
        float avg_time = float(time) / float(exec_times);
        std::cout << "<customer bugs case time cost: "
                  << avg_time / 1000 << +"ms>" << std::endl;
  }

  void ShowAverageBatchTime(int exec_times, int call_times) {
        size_t time = GetTotalMicroseconds();
        float avg_time = float(time) / float(exec_times) / float(call_times);
        std::cout << "per batch time cost: "
                  << avg_time / 1000 << +"ms" << std::endl;
  }

  void ShowMinMaxTime(int exec_times) {
		std::cout << "MinMax on "<< exec_times << " runs - " << "min/max time cost: "
				<< float(min_us_) / 1000 << +"ms / "
				<< float(max_us_) / 1000 << +"ms" << std::endl;
  }
};
}
#endif /* UTIL_H_ */

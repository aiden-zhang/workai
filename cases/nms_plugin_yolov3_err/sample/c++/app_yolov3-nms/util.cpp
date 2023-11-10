#include "util.h"

bool performance_test = false;

void SetPerformanceMode(bool mode) {
	performance_test = mode;
}

namespace util {
void cuda_check(cudaError_t state, std::string file, int line) {
  if (state != cudaSuccess) {
    std::cout << "CUDA Error code num is:" << state << std::endl;
    std::cout << "CUDA Error:" << cudaGetErrorString(state) << std::endl;
    std::cout << file << " " << line << "line!" << std::endl;
    ASSERT(false);
  }
}
}

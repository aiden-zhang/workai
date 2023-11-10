






#ifndef DL_ERROR_CHECK_H
#define DL_ERROR_CHECK_H

#include <iostream>

#include "cuda.h"
#include "cudnn.h"

#define ASSERT(x)      \
  do {                 \
    if (!(x)) abort(); \
  } while (0)

#define CUDA_CHECK(x) cuda_check(x, __FILE__, __LINE__)
#define CUDNN_CHECK(x) cudnn_check(x, __FILE__, __LINE__)

void cuda_check(cudaError_t state, std::string file, int line);
void cudnn_check(cudnnStatus_t state, std::string file, int line);

#endif //DL_ERROR_CHECK_H

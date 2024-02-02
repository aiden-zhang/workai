#ifndef RGB_ROTATE_H_
#define RGB_ROTATE_H_

#include <stdio.h>
#include "common.h"

#define TILE_HW  32

template <typename T>
__global__ void rgb_rotate_180(T * __restrict__ in, T * __restrict__ out,
        int in_h, int in_w) {
  int in_x = threadIdx.x + blockIdx.x * blockDim.x; // w
  int in_y = threadIdx.y + blockIdx.y * blockDim.y; // h
  int out_x = in_w - 1 - in_x; // w
  int out_y = in_h - 1 - in_y; // h
  if (in_x < in_w && in_y < in_h)
  {
    out[out_y * in_w + out_x] = in[in_y * in_w + in_x];
  }
}

template <typename T>
__global__ void rgb_rotate_90(T * __restrict__ in, T * __restrict__ out, int in_h, int in_w) {
  __shared__ T tile[TILE_HW][TILE_HW];
  int in_x = threadIdx.x + blockIdx.y * blockDim.y; //w
  int in_y = threadIdx.y + blockIdx.x * blockDim.x; //h
  int out_x = threadIdx.x + blockIdx.x * blockDim.x; //h
  int out_y = threadIdx.y + blockIdx.y * blockDim.y; //w
  if (in_x < in_w && in_y < in_h) {
    tile[threadIdx.x][threadIdx.y] = in[(in_h - 1 - in_y) * in_w + in_x];
  }
  __syncthreads();
  if (out_x < in_h && out_y < in_w) {
    out[out_y * in_h + out_x] = tile[threadIdx.y][threadIdx.x];
  }
}

template <typename T>
__global__ void rgb_rotate_270(T * __restrict__ in, T * __restrict__ out, int in_h, int in_w) {
  __shared__ T tile[TILE_HW][TILE_HW];
  int in_x = threadIdx.x + blockIdx.y * blockDim.y; //w
  int in_y = threadIdx.y + blockIdx.x * blockDim.x; //h
  int out_x = threadIdx.x + blockIdx.x * blockDim.x; //h
  int out_y = threadIdx.y + blockIdx.y * blockDim.y; //w
  if (in_x < in_w && in_y < in_h) {
    tile[threadIdx.x][threadIdx.y] = in[in_y * in_w + in_w - 1 - in_x];
  }
  __syncthreads();
  if (out_x < in_h && out_y < in_w) {
    out[out_y * in_h + out_x] = tile[threadIdx.y][threadIdx.x];
  }
}

#endif

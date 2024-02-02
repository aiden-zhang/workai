#ifndef YUV_CROP_H_
#define YUV_CROP_H_

#include <stdio.h>
#include "common.h"

template <typename TY, typename TUV>
__global__ void yu12_crop_kernel(TY *__restrict__ in_y, TUV *__restrict__ in_u, TUV *__restrict__ in_v,
                                 TY *__restrict__ out_y, TUV *__restrict__ out_u, TUV *__restrict__ out_v,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w)
{
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (h_idx < out_h && x_idx < out_w)
  {
    if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w)
    {
      out_y[h_idx * out_w + x_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
    }
  }

  int in_uv_h = in_h >> 1;
  int out_uv_h = out_h >> 1;
  if (h_idx < out_uv_h && x_idx < out_w)
  {
    if (((start_h >> 1) + h_idx < in_uv_h) && (start_w + x_idx) < in_w)
    {
      int in_idx = ((start_h >> 1) + h_idx) * in_w + start_w + x_idx;
      int out_idx = h_idx * out_w + x_idx;
      out_u[out_idx] = in_u[in_idx];
      out_v[out_idx] = in_v[in_idx];
    }
  }
}

template <typename T>
__global__ void nv12_crop_kernel(T *__restrict__ in_y, T *__restrict__ in_uv,
                                 T *__restrict__ out_y, T *__restrict__ out_uv,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w)
{
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (h_idx < out_h && x_idx < out_w)
  {
    if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w)
    {
      out_y[h_idx * out_w + x_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
    }
  }

  int in_uv_h = in_h >> 1;
  int out_uv_h = out_h >> 1;
  if (h_idx < out_uv_h && x_idx < out_w)
  {
    if (((start_h >> 1) + h_idx < in_uv_h) && (start_w + x_idx) < in_w)
    {
      out_uv[h_idx * out_w + x_idx] = in_uv[((start_h >> 1) + h_idx) * in_w + start_w + x_idx];
    }
  }
}

__global__ void yu12_crop_kernel(uint8_t *__restrict__ in_y, uint8_t *__restrict__ in_u, uint8_t *__restrict__ in_v,
                                 uint8_t *__restrict__ out_y, uint8_t *__restrict__ out_u, uint8_t *__restrict__ out_v,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w)
{
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (h_idx < out_h && x_idx < out_w)
  {
    if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w)
    {
      out_y[h_idx * out_w + x_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
    }
  }

  int in_uv_h = in_h >> 1;
  int in_uv_w = in_w >> 1;
  int out_uv_h = out_h >> 1;
  int out_uv_w = out_w >> 1;
  if (h_idx < out_uv_h && x_idx < out_uv_w)
  {
    if (((start_h >> 1) + h_idx < in_uv_h) && (start_w >> 1) + x_idx < in_uv_w)
    {
      int in_idx = ((start_h >> 1) + h_idx) * in_uv_w + (start_w >> 1) + x_idx;
      int out_idx = h_idx * out_uv_w + x_idx;
      out_u[out_idx] = in_u[in_idx];
      out_v[out_idx] = in_v[in_idx];
    }
  }
}

__global__ void nv12_crop_kernel(uint8_t *__restrict__ in_y, uint8_t *__restrict__ in_uv,
                                 uint8_t *__restrict__ out_y, uint8_t *__restrict__ out_uv,
                                 int start_h, int start_w, int in_h, int in_w, int out_h, int out_w)
{
  int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int h_idx = threadIdx.y + blockIdx.y * blockDim.y;

  if (h_idx < out_h && x_idx < out_w)
  {
    if ((start_h + h_idx) < in_h && (start_w + x_idx) < in_w)
    {
      int out_idx = h_idx * out_w + x_idx;
      out_y[out_idx] = in_y[(start_h + h_idx) * in_w + start_w + x_idx];
      if (h_idx < (out_h >> 1))
      {
        out_uv[out_idx] = in_uv[((start_h >> 1) + h_idx) * in_w + start_w + x_idx];
      }
    }
  }
}

#endif


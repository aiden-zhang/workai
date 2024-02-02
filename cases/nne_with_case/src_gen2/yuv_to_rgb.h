#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "common.h"


__device__ __forceinline__ uint8_t clip(float value) {
  value += 0.5f;
  value = min(max(0.f, value), 255.f);
  return value;
}

__device__ __forceinline__ uchar3 convert2rgb_TV_range(float y, float u, float v) {
    uchar3 tmp;
    y -= 16.f;
    y = max(0.f, y);
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(1.164f * y + 1.596f * v);
    tmp.y = clip(1.164f * y - 0.813f * v - 0.391f * u);
    tmp.z = clip(1.164f * y + 2.018f * u);
    return tmp;
}

template <typename OUT, bool plane, int tile_size>
__global__ void YU122rgb24_general(const uint8_t *__restrict__ in_Y,
                                   const uint8_t *__restrict__ in_U,
                                   const uint8_t *__restrict__ in_V,
                                   OUT *__restrict__ out, int32_t h, int w)
{
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int uv_h_idx = h_idx >> 1;
    int uv_stride = w >> 1;
    if (w_idx < w / tile_size && h_idx < h)
    {
        for (int l = 0; l < tile_size; l++)
        {
            int i = w_idx * tile_size + l;
            int uv_w_idx = i >> 1;
            float y = in_Y[h_idx * w + i];
            float u = in_U[uv_h_idx * uv_stride + uv_w_idx];
            float v = in_V[uv_h_idx * uv_stride + uv_w_idx];
            uchar3 rgb = convert2rgb_TV_range(y, u, v);
            int idx = h_idx * w + i;
            if (plane)
            {
                out[idx] = rgb.x;
                out[idx + h * w] = rgb.y;
                out[idx + h * w * 2] = rgb.z;
            }
            else
            {
                out[idx * 3 + 0] = rgb.x;
                out[idx * 3 + 1] = rgb.y;
                out[idx * 3 + 2] = rgb.z;
            }
        }
    }
    if (tile_size > 1)
    {
        int remain = w & (tile_size - 1);
        if (w_idx < remain && h_idx < h)
        {
            int i = w - remain + w_idx;
            int uv_w_idx = i >> 1;
            float y = in_Y[h_idx * w + i];
            float u = in_U[uv_h_idx * uv_stride + uv_w_idx];
            float v = in_V[uv_h_idx * uv_stride + uv_w_idx];
            uchar3 rgb = convert2rgb_TV_range(y, u, v);
            int idx = h_idx * w + i;
            if (plane)
            {
                out[idx] = rgb.x;
                out[idx + h * w] = rgb.y;
                out[idx + h * w * 2] = rgb.z;
            }
            else
            {
                out[idx * 3 + 0] = rgb.x;
                out[idx * 3 + 1] = rgb.y;
                out[idx * 3 + 2] = rgb.z;
            }
        }
    }
}

template <typename OUT, bool plane, int tile_size>
__global__ void NV122rgb24_general(const uint8_t *__restrict__ in_Y,
                                   const uint8_t *__restrict__ in_UV,
                                   OUT *__restrict__ out, int32_t h, int w)
{
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (w_idx < w / tile_size && h_idx < h)
    {
        for (int l = 0; l < tile_size; l++)
        {
            int i = w_idx * tile_size + l;
            int uv_h_idx = h_idx >> 1;
            int uv_w_idx = i & 0xfffffffe;
            float y = in_Y[h_idx * w + i];
            float u = in_UV[uv_h_idx * w + uv_w_idx];
            float v = in_UV[uv_h_idx * w + uv_w_idx + 1];
            uchar3 rgb = convert2rgb_TV_range(y, u, v);
            int idx = h_idx * w + i;
            if (plane)
            {
                out[idx] = rgb.x;
                out[idx + h * w] = rgb.y;
                out[idx + h * w * 2] = rgb.z;
            }
            else
            {
                out[idx * 3 + 0] = rgb.x;
                out[idx * 3 + 1] = rgb.y;
                out[idx * 3 + 2] = rgb.z;
            }
        }
    }
    if (tile_size > 1)
    {
        int remain = w & (tile_size - 1);
        if (w_idx < remain && h_idx < h)
        {
            int i = w - remain + w_idx;
            int uv_h_idx = h_idx >> 1;
            int uv_w_idx = i & 0xfffffffe;
            float y = in_Y[h_idx * w + i];
            float u = in_UV[uv_h_idx * w + uv_w_idx];
            float v = in_UV[uv_h_idx * w + uv_w_idx + 1];
            uchar3 rgb = convert2rgb_TV_range(y, u, v);
            int idx = h_idx * w + i;
            if (plane)
            {
                out[idx] = rgb.x;
                out[idx + h * w] = rgb.y;
                out[idx + h * w * 2] = rgb.z;
            }
            else
            {
                out[idx * 3 + 0] = rgb.x;
                out[idx * 3 + 1] = rgb.y;
                out[idx * 3 + 2] = rgb.z;
            }
        }
    }
}

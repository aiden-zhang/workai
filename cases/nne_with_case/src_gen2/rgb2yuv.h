#ifndef RGB_TO_YUV_H_
#define RGB_TO_YUV_H_

#include <stdio.h>
#include "common.h"


__device__ uchar3 rgb2yuv(uint8_t r, uint8_t g, uint8_t b) {
    uchar3 temp;
    //temp.x = clip(0.256999969f * r + 0.50399971f * g + 0.09799957f * b);
    //temp.y = clip(-0.1479988098f * r + -0.2909994125f * g + 0.438999176f * b  + 128.f);
    //temp.z = clip(0.438999176f * r + -0.3679990768f * g + -0.0709991455f * b  + 128.f);
    temp.x = (uint8_t)((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    temp.y = (uint8_t)((-38 * r -74 * g + 112 * b  + 128) >> 8) + 128;
    temp.z = (uint8_t)((112 * r - 94 * g - 18 * b  + 128) >> 8) + 128;
    return temp;
}
__device__ uint8_t rgb2y(int32_t r, int32_t g, int32_t b) {
    return (uint8_t)((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
}
__device__ uint8_t rgb2u(int32_t r, int32_t g, int32_t b) {
    return (uint8_t)((-38 * r -74 * g + 112 * b  + 128) >> 8) + 128;
}
__device__ uint8_t rgb2v(int32_t r, int32_t g, int32_t b) {
    return (uint8_t)((112 * r - 94 * g - 18 * b  + 128) >> 8) + 128;
}

__global__ void rgb2yu12_kernel(uchar3 *in, uint8_t *out_y, uint8_t *out_u, uint8_t *out_v, int w, int h)
{
    int h_idx = blockIdx.y * 2;
    int in_offset0 = h_idx * w;
    int in_offset1 = in_offset0 + w; 
    int half_w = w >> 1;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < half_w) {
        uchar2 y1, y2;
        uchar3 a, b, c, d;
        a = in[in_offset0 + w_idx * 2  + 0];
        b = in[in_offset0 + w_idx * 2  + 1];
        c = in[in_offset1 + w_idx * 2  + 0];
        d = in[in_offset1 + w_idx * 2  + 1];
        y1.x = rgb2y(a.x, a.y, a.z);
        y1.y = rgb2y(b.x, b.y, b.z);
        int out_idx = blockIdx.y * half_w + w_idx;
        out_u[out_idx] = rgb2u(a.x, a.y, a.z);
        out_v[out_idx] = rgb2v(a.x, a.y, a.z);
        y2.x = rgb2y(c.x, c.y, c.z);
        y2.y = rgb2y(d.x, d.y, d.z);
        out_idx = h_idx * w + w_idx * 2;
        out_y[out_idx] = y1.x;
        out_y[out_idx + 1] = y1.y;
        out_y[out_idx + w] = y2.x;
        out_y[out_idx + w + 1] = y2.y;
    }
}

#endif

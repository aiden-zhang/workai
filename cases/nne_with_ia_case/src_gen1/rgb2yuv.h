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

template<bool align = false>
__global__ void rgb2yu12_kernel(uint8_t *in, uint8_t *out, int w, int h) {
    extern __shared__ uint8_t sm[];
    int h_idx = blockIdx.x * 2;
    uint8_t *h1 = sm;
    uchar2 *out_y = (uchar2*)out;
    uint8_t *out_u = out + h * w;
    uint8_t *out_v = out_u + (h * w >> 2);
    if (align) {
        global2share_copy_align(in +  h_idx * w * 3, h1, w * 6);
    } else {
        int offset = global2share_copy(in + h_idx * w * 3, h1, w * 6);
        h1 += offset;
    }
    __syncthreads();
    uint8_t *h2 = h1 + w * 3;
    
    int half_w = w >> 1;
    for (int w_idx = threadIdx.x; w_idx < half_w; w_idx += blockDim.x) {
        uchar2 y1, y2;
        int offset = w_idx * 6;
        int32_t r = h1[offset + 0];
        int32_t g = h1[offset + 1];
        int32_t b = h1[offset + 2];
        y1.x = rgb2y(r, g, b);
        int out_idx = blockIdx.x * half_w + w_idx;
        out_u[out_idx] = rgb2u(r, g, b);
        out_v[out_idx] = rgb2v(r, g, b);
        r = h2[offset + 0];
        g = h2[offset + 1];
        b = h2[offset + 2];
        y2.x = rgb2y(r, g, b);
        r = h1[offset + 3];
        g = h1[offset + 4];
        b = h1[offset + 5];
        y1.y = rgb2y(r, g, b);
        r = h2[offset + 3];
        g = h2[offset + 4];
        b = h2[offset + 5];
        y2.y = rgb2y(r, g, b);
        out_idx = h_idx * half_w + w_idx;
        out_y[out_idx] = y1;
        out_y[out_idx + half_w] = y2;
    }
}

#endif

#ifndef RGB_CROP_H_
#define RGB_CROP_H_

#include <stdio.h>
#include "common.h"

__global__ void rgb_crop_kernel(uchar3 *__restrict__ input, uchar3 *__restrict__ output,
    int start_h, int start_w, int in_h, int in_w, int out_h, int out_w) {
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = blockIdx.y;
    if (w_idx < out_w) {
        output[h_idx * out_w + w_idx] = input[(start_h + h_idx) * in_w + start_w + w_idx];
    }
}

#endif


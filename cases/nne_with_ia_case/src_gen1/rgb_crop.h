#ifndef RGB_CROP_H_
#define RGB_CROP_H_

#include <stdio.h>
#include "common.h"

__global__ void rgb_crop_kernel(uint8_t *__restrict__ input,
                                    uint8_t *__restrict__ output,
                                    int start_h, int start_w, int in_h, int in_w, int out_h, int out_w, int c) {
    int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (h_idx < out_h) {
        if (x_idx < out_w * c) {
            output[h_idx * out_w * c + x_idx] = input[(start_h + h_idx) * in_w * c + start_w * c + x_idx];
        }
    }
}

#endif


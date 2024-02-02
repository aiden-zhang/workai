#ifndef RGB_NORMALIZATION_H_
#define RGB_NORMALIZATION_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

template<typename IN, typename OUT>
__global__ void rgb_normalization_kernel(IN *__restrict__ input,
                                    OUT *__restrict__ output,
                                    float mean, 
                                    float std,
                                    float scale,
                                    int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        output[idx] = (input[idx] * scale - mean) * std;
    }
}

template<typename IN, typename OUT, bool input_plane, bool output_plane>
__global__ void rgb_normalization_3channels_kernel(IN *__restrict__ input,
                                    OUT *__restrict__ output,
                                    float mean1, 
                                    float mean2, 
                                    float mean3, 
                                    float std1,
                                    float std2,
                                    float std3,
                                    float scale,
                                    int w,
                                    int h,
                                    int channel_rev) {
    int h_idx = blockIdx.y;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < w)
    {
        int in_offset0, in_offset1, in_offset2;
        if (input_plane)
        {
            in_offset0 = h_idx * w + 0 * w * h + w_idx;
            in_offset1 = h_idx * w + 1 * w * h + w_idx;
            in_offset2 = h_idx * w + 2 * w * h + w_idx;
        }
        else
        {
            in_offset0 = (h_idx * w + w_idx) * 3 + 0;
            in_offset1 = (h_idx * w + w_idx) * 3 + 1;
            in_offset2 = (h_idx * w + w_idx) * 3 + 2;
        }
        int out_offset0, out_offset1, out_offset2;
        if (output_plane)
        {
            out_offset0 = channel_rev * w * h + h_idx * w + w_idx;
            out_offset1 = 1 * w * h + h_idx * w + w_idx;
            out_offset2 = (2 - channel_rev) * w * h + h_idx * w + w_idx;
        }
        else
        {
            out_offset0 = (h_idx * w + w_idx) * 3 + channel_rev;
            out_offset1 = (h_idx * w + w_idx) * 3 + 1;
            out_offset2 = (h_idx * w + w_idx) * 3 + (2 - channel_rev);
        }
        output[out_offset0] = (input[in_offset0] * scale - mean1) * std1;
        output[out_offset1] = (input[in_offset1] * scale - mean2) * std2;
        output[out_offset2] = (input[in_offset2] * scale - mean3) * std3;
    }
}
#endif

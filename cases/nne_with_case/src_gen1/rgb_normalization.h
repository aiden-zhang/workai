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

template<typename IN, typename OUT, bool input_plane, bool output_plane, bool align = false, int channel_rev = 0>
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
                                    int h) {
    extern __shared__ uint8_t sm[];
    uint8_t * sm_line = sm;
    uint8_t *h1 = sm;
    uint8_t *h2 = h1 + (w & 0xfffffff8) + 16;
    uint8_t *h3 = h2 + (w & 0xfffffff8) + 16;
    int h_idx = blockIdx.x;
    if (input_plane) {
        if (align) {
            global2share_copy_align(input +  h_idx * w + 0 * w * h, h1, w);
            global2share_copy_align(input +  h_idx * w + 1 * w * h, h2, w);
            global2share_copy_align(input +  h_idx * w + 2 * w * h, h3, w);
        } else {
            int offset = global2share_copy(input + h_idx * w + 0 * w * h, h1, w);
            h1 += offset;
            offset = global2share_copy(input + h_idx * w + 1 * w * h, h2, w);
            h2 += offset;
            offset = global2share_copy(input + h_idx * w + 2 * w * h, h3, w);
            h3 += offset;
        }
    } else {
        if (align) {
            global2share_copy_align(input +  h_idx * w * 3, sm_line, w * 3);
        } else {
            int offset = global2share_copy(input + h_idx * w * 3, sm_line, w * 3);
            sm_line += offset;
        }
    }
    __syncthreads();

    if (output_plane) {
        if (input_plane) {
            for (int w_idx = threadIdx.x; w_idx < w; w_idx += blockDim.x) {
                int idx = h_idx * w + w_idx;
                output[idx + channel_rev * w * h] = (h1[w_idx] * scale - mean1) * std1; 
                output[idx + 1 * w * h] = (h2[w_idx] * scale - mean2) * std2; 
                output[idx + (2 - channel_rev) * w * h] = (h3[w_idx] * scale - mean3) * std3; 
            }
        } else {
            for (int w_idx = threadIdx.x; w_idx < w; w_idx += blockDim.x) {
                int idx = h_idx * w + w_idx;
                output[idx + channel_rev * w * h] = (sm_line[w_idx * 3 + 0] * scale - mean1) * std1; 
                output[idx + 1 * w * h] = (sm_line[w_idx * 3 + 1] * scale - mean2) * std2; 
                output[idx + (2 - channel_rev) * w * h] = (sm_line[w_idx * 3 + 2] * scale - mean3) * std3; 
            }
        }
    } else {
        if (input_plane) {
            for (int w_idx = threadIdx.x; w_idx < w; w_idx += blockDim.x) {
                int idx = h_idx * w + w_idx;
                output[idx * 3 + channel_rev] = (h1[w_idx] * scale - mean1) * std1; 
                output[idx * 3 + 1] = (h2[w_idx] * scale - mean2) * std2; 
                output[idx * 3 + (2 - channel_rev)] = (h3[w_idx] * scale - mean3) * std3; 
            }
        } else {
            for (int w_idx = threadIdx.x; w_idx < w; w_idx += blockDim.x) {
                int idx = h_idx * w + w_idx;
                output[idx * 3 + channel_rev] = (sm_line[w_idx * 3 + 0] * scale - mean1) * std1; 
                output[idx * 3 + 1] = (sm_line[w_idx * 3 + 1] * scale - mean2) * std2; 
                output[idx * 3 + (2 - channel_rev)] = (sm_line[w_idx * 3 + 2] * scale - mean3) * std3; 
            }
        }
    }
}
#endif

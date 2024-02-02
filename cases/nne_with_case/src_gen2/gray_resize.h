#ifndef RGB_RESIZE_H_
#define RGB_RESIZE_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

// input and output format is HW
template<bool isBilinear = true>
__global__  void gray_resize_kernel(const uint8_t *__restrict__ input,
        int h, int w, int t_h, int t_w, float t_h_trans, float t_w_trans,
        uint8_t *__restrict__ out) {
    int h_idx = blockIdx.y;
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (!isBilinear) {
        int raw_h_idx = h_idx * t_h_trans;
        int raw_w_idx = w_idx * t_w_trans;
        int out_idx = h_idx * t_w + w_idx;
        int in_idx = raw_h_idx * w + raw_w_idx;
        if (w_idx < t_w) {
            out[out_idx] = input[in_idx]; 
        }
    } else {
        float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
        int top = floor(raw_h_idx);
        int bottom = MIN(h - 1, top + 1);
        __half diff = __float2half_rn(raw_h_idx - top);
        top = MAX(top, 0);
        if (w_idx < t_w) {
            float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
            int left = raw_w_idx;
            int right = MIN(w - 1, left + 1);
            half2 w_bi, alpha_h;
            alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left);
            left = MAX(left, 0);

            half2 x1, x2;
            x1.x = input[top * w + left];
            x2.x = input[top * w + right];
            x1.y = input[bottom * w + left];
            x2.y = input[bottom * w + right];
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out[h_idx * t_w + w_idx] =
                __half2int_rn((w_bi.y - w_bi.x) * diff + w_bi.x);
        }
    }
}

#endif

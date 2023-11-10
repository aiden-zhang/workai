#ifndef RGB_RESIZE_H_
#define RGB_RESIZE_H_

#include <stdio.h>

#include "cuda_fp16.h"
#include "common.h"

// input and output format is HW
template<bool isBilinear = true, bool align = true>
__global__  void gray_resize_kernel(const uint8_t *__restrict__ input,
        int h, int w, int t_h, int t_w, float t_h_trans, float t_w_trans,
        uint8_t *__restrict__ out) {
    extern __shared__ uint8_t sm[];
    uint8_t * sm_line = sm;
    int h_idx = blockIdx.x;
    if(h_idx < t_h) {
        if (!isBilinear) {
            int raw_h_idx = h_idx * t_h_trans;
            if (align) {
                global2share_copy_align(input + raw_h_idx * w, sm_line, w);
            } else {
                int offset = global2share_copy(input + raw_h_idx * w, sm_line, w);
                sm_line += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                int raw_w_idx = w_idx * t_w_trans;
                out[h_idx * t_w + w_idx] = sm_line[raw_w_idx]; 
            }
        } else {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
            int top_h    = raw_h_idx;
            int line_id = 0;
            int copy_size = w;
            if (top_h + 1 < h) {
                copy_size *= 2;
                line_id = 1;
            }
            if (align) {
                global2share_copy_align(input + top_h * w, sm_line, copy_size);
            } else {
                int offset = global2share_copy(input + top_h * w, sm_line, copy_size);
                sm_line += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
                int left_w   = raw_w_idx;
                int right_w  = MIN(w - 1, left_w + 1);
                half2 x1, x2;
                x1.x = __ushort2half_rn(sm_line[left_w]);
                x2.x = __ushort2half_rn(sm_line[right_w]);
                x1.y = __ushort2half_rn(sm_line[left_w  + w * line_id]);
                x2.y = __ushort2half_rn(sm_line[right_w + w * line_id]);

                half2 w_bi, alpha_h;
                alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left_w);
                w_bi = __hsub2(x2, x1);
                w_bi = __hfma2(w_bi, alpha_h, x1);
                out[h_idx * t_w + w_idx] =
                    __half2int_rn(
                            (w_bi.y - w_bi.x) * __float2half_rn(raw_h_idx - top_h)
                            + w_bi.x);
            }
        }
    }
}

#endif

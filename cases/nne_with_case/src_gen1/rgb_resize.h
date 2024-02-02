#ifndef RGB_RESIZE_H_
#define RGB_RESIZE_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

// input and output format is NHWC
template<bool isBilinear = true, int c = 3, bool align = true>
__global__  void rgb_resize_kernel(const uint8_t *__restrict__ input,
                                  int h,
                                  int w,
                                  int t_h,
                                  int t_w,
                                  float t_h_trans,
                                  float t_w_trans,
                                  uint8_t *__restrict__ out) {
    extern __shared__ uint8_t sm[];
    uint8_t * sm_line = sm;
    int h_idx = blockIdx.x;
    if (h_idx < t_h) {
        if (!isBilinear) {
            int raw_h_idx = h_idx * t_h_trans;
            if (align) {
                global2share_copy_align(input + raw_h_idx * w * c, sm_line, w * c);
            } else {
                int offset = global2share_copy(input + raw_h_idx * w * c, sm_line, w * c);
                sm_line += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                int raw_w_idx = w_idx * t_w_trans;
                for (int c_idx = 0; c_idx < c; c_idx++) {
                    out[(h_idx * t_w + w_idx) * c + c_idx] = sm_line[raw_w_idx * c + c_idx];
                }
            }
        } else {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
            int top_h = raw_h_idx;
            int line_id = 0;
            int copy_size = w * c;
            if (top_h + 1 < h) {
                copy_size *= 2;
                line_id = 1;
            } 
            if (align) {
                global2share_copy_align(input + top_h * w * c, sm_line, copy_size);
            } else {
                int offset = global2share_copy(input + top_h * w * c, sm_line, copy_size);
                sm_line += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                for (int c_idx = 0; c_idx < c; c_idx++) {
                    float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
                    int left_w = raw_w_idx;
                    int right_w = MIN(w - 1, left_w + 1);
                    half2 x1, x2;
                    x1.x = __ushort2half_rn(sm_line[c_idx + left_w * c]);
                    x2.x = __ushort2half_rn(sm_line[c_idx + right_w * c]);
                    x1.y = __ushort2half_rn(sm_line[c_idx + (w * line_id + left_w) * c]);
                    x2.y = __ushort2half_rn(sm_line[c_idx + (w * line_id + right_w) * c]);
                    half2 w_bi, alpha_h;
                    alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left_w);
                    w_bi = __hsub2(x2, x1);
                    w_bi = __hfma2(w_bi, alpha_h, x1);
                    out[(h_idx * t_w + w_idx) * c + c_idx] =
                        __half2int_rn(
                                (w_bi.y - w_bi.x) * __float2half_rn(raw_h_idx - top_h)
                                + w_bi.x);
                }
            }
        }
    }
}


template<bool isBilinear = true, bool align = true>
__global__  void rgb_plane_resize_pad_kernel(const uint8_t *__restrict__ input,
                                  int h,
                                  int w,
                                  int t_h,
                                  int t_w,
                                  int out_h,
                                  int out_w,
                                  int b_h,
                                  int b_w,
                                  float t_h_trans,
                                  float t_w_trans,
                                  uint8_t *__restrict__ out) {
    extern __shared__ uint8_t sm[];
    int h_idx = blockIdx.x;
    if (h_idx < t_h) {
        if (!isBilinear) {
            int raw_h_idx = h_idx * t_h_trans;
            for (int c_idx = 0; c_idx < 3; ++c_idx) {
                uint8_t * sm_line = sm;
                if (align) {
                    global2share_copy_align(input + raw_h_idx * w + c_idx * h * w, sm_line, w);
                } else {
                    int offset = global2share_copy(input + raw_h_idx * w + c_idx * h * w, sm_line, w);
                    sm_line += offset;
                }
                __syncthreads();
                int out_idx = (h_idx + b_h) * out_w + b_w + c_idx * out_h * out_w;
                for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                    int raw_w_idx = w_idx * t_w_trans;
                    out[out_idx + w_idx] = sm_line[raw_w_idx];
                }
                __syncthreads();
            }
        } else {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
            int top_h = raw_h_idx;
            int line_id = 0;
            int copy_size = w;
            if (top_h + 1 < h) {
                copy_size *= 2;
                line_id = 1;
            } 
            for (int c_idx = 0; c_idx < 3; ++c_idx) {
                uint8_t * sm_line = sm;
                if (align) {
                    global2share_copy_align(input + top_h * w + c_idx * h * w, sm_line, copy_size);
                } else {
                    int offset = global2share_copy(input + top_h * w + c_idx * h * w, sm_line, copy_size);
                    sm_line += offset;
                }
                __syncthreads();
                int out_idx = (h_idx + b_h) * out_w + b_w + c_idx * out_h * out_w;
                for (int w_idx = threadIdx.x; w_idx < t_w; w_idx += blockDim.x) {
                    float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
                    int left_w = raw_w_idx;
                    int right_w = MIN(w - 1, left_w + 1);
                    half2 x1, x2;
                    x1.x = sm_line[left_w];
                    x2.x = sm_line[right_w];
                    x1.y = sm_line[w * line_id + left_w];
                    x2.y = sm_line[w * line_id + right_w];
                    half2 w_bi, alpha_h;
                    alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left_w);
                    w_bi = __hsub2(x2, x1);
                    w_bi = __hfma2(w_bi, alpha_h, x1);
                    out[out_idx + w_idx] =__half2int_rn(
                                (w_bi.y - w_bi.x) * __float2half_rn(raw_h_idx - top_h) + w_bi.x);
                }
                __syncthreads();
            }
        }
    }
}

template<bool isBilinear = true, int c = 3, bool align = true>
__global__  void rgb_resize_ROI_kernel(const uint8_t *__restrict__ input,
        int h,
        int w,
        int h_out,
        int w_out,
        float t_h_trans,
        float t_w_trans,
        int roi_h_start,
        int roi_w_start,
        int h_roi,
        int w_roi,
        uint8_t *__restrict__ out) {
    extern __shared__ uint8_t sm[];
    uint8_t * sm_line = sm;
    int h_idx = blockIdx.x;
    if (h_idx < h_out) {
        if (!isBilinear) {
            int raw_h_idx = h_idx * t_h_trans + roi_h_start;
            if (align) {
                global2share_copy_align(input + raw_h_idx * w * c + roi_w_start * c, sm_line, w_roi * c);
            } else {
                int offset = global2share_copy(input + raw_h_idx * w * c + roi_w_start * c, sm_line, w_roi * c);
                sm_line += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < w_out; w_idx += blockDim.x) {
                for (int c_idx = 0; c_idx < c; ++c_idx) {
                    int raw_w_idx = w_idx * t_w_trans;
                    out[(h_idx * w_out + w_idx) * c + c_idx]
                        = sm_line[raw_w_idx * c + c_idx];
                }
            }
        } else {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f + roi_h_start;
            int top_h = raw_h_idx;
            int bottom_h = MIN(h - 1, top_h + 1);
            int copy_size = w_roi * c;
            int line_id = top_h + 1 < h ? 1 : 0;
            uint8_t *sm_h1 = sm_line;
            uint8_t *sm_h2 = sm_line + ((copy_size + 16) & 0xfffffff8); 
            if (align) {
                global2share_copy_align(input + (top_h * w + roi_w_start) * c, sm_h1, copy_size);
                global2share_copy_align(input + (bottom_h * w + roi_w_start) * c, sm_h2, copy_size);
            } else {
                int offset = global2share_copy(input + (top_h * w + roi_w_start) * c, sm_h1, copy_size);
                sm_h1 += offset;
                offset = global2share_copy(input + (bottom_h * w + roi_w_start) * c, sm_h2, copy_size);
                sm_h2 += offset;
            }
            __syncthreads();
            for (int w_idx = threadIdx.x; w_idx < w_out; w_idx += blockDim.x) {
                for (int c_idx = 0; c_idx < c; ++c_idx) {
                    float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
                    int left_w = raw_w_idx;
                    int right_w = MIN(w - roi_w_start - 1, left_w + 1);
                    half2 x1, x2;
                    x1.x = __ushort2half_rn(sm_h1[c_idx + left_w * c]);
                    x2.x = __ushort2half_rn(sm_h1[c_idx + right_w * c]);
                    x1.y = __ushort2half_rn(sm_h2[c_idx + left_w * c]);
                    x2.y = __ushort2half_rn(sm_h2[c_idx + right_w * c]);
                    half2 w_bi, alpha_h;
                    alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left_w);
                    w_bi = __hsub2(x2, x1);
                    w_bi = __hfma2(w_bi, alpha_h, x1);
                    out[(h_idx * w_out + w_idx) * c + c_idx] =
                        __half2int_rn(
                                (w_bi.y - w_bi.x) * __float2half_rn(raw_h_idx - top_h)
                                + w_bi.x);
                }
            }
        }
    }
}
#endif

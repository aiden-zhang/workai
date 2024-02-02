#ifndef RGB_RESIZE_H_
#define RGB_RESIZE_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

// input and output format is NHWC
template<bool isBilinear = true>
__global__  void rgb_resize_kernel(const uchar3 *__restrict__ input,
        int h, int w, int t_h, int t_w, float t_h_trans, float t_w_trans, uchar3 *__restrict__ out) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (!isBilinear) {
        for (; idx < t_w * t_h; idx += blockDim.x * gridDim.x)
        {
            int h_idx = idx / t_w;
            int w_idx = idx % t_w;
            int raw_h_idx = h_idx * t_h_trans;
            int raw_w_idx = w_idx * t_w_trans;
            out[h_idx * t_w + w_idx] = input[raw_h_idx * w + raw_w_idx];
        }
    } else {
        for (; idx < t_w * t_h; idx += blockDim.x * gridDim.x)
        {
            int h_idx = idx / t_w;
            int w_idx = idx % t_w;
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
            float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
            int top = raw_h_idx;
            int bottom = MIN(h - 1, top + 1);
            __half diff_y = __float2half_rn(raw_h_idx - top);
            top = MAX(top, 0);
            int left = raw_w_idx;
            int right = MIN(w - 1, left + 1);
            half2 w_bi, alpha_h;
            alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left);
            left = MAX(left, 0);
            uchar3 a, b, c, d, out_tmp;
            a = input[top * w + left];
            b = input[top * w + right];
            c = input[bottom * w + left];
            d = input[bottom * w + right];
            half2 x1, x2;
            
            x1.x = a.x; x2.x = b.x; x1.y = c.x; x2.y = d.x;
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_tmp.x = __half2int_rn((w_bi.y - w_bi.x) * diff_y + w_bi.x);
            x1.x = a.y; x2.x = b.y; x1.y = c.y; x2.y = d.y;
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_tmp.y = __half2int_rn((w_bi.y - w_bi.x) * diff_y + w_bi.x);
            x1.x = a.z; x2.x = b.z; x1.y = c.z; x2.y = d.z;
            w_bi = __hsub2(x2, x1);
            w_bi = __hfma2(w_bi, alpha_h, x1);
            out_tmp.z = __half2int_rn((w_bi.y - w_bi.x) * diff_y + w_bi.x);

            out[h_idx * t_w + w_idx] = out_tmp;
        }
    }
}


template<bool isBilinear = true>
__global__  void rgb_plane_resize_pad_kernel(const uint8_t *__restrict__ input,
        int h, int w, int t_h, int t_w, int out_h, int out_w, int b_h,
        int b_w, float t_h_trans, float t_w_trans, uint8_t *__restrict__ out) {
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (!isBilinear) {
        if (w_idx < t_w && h_idx < t_h)
        {
            int raw_h_idx = h_idx * t_h_trans;
            int raw_w_idx = w_idx * t_w_trans;
            int out_idx = (h_idx + b_h) * out_w + b_w + w_idx;
            out[out_idx + 0 * out_h * out_w] = input[raw_h_idx * w + raw_w_idx + 0 * h * w];
            out[out_idx + 1 * out_h * out_w] = input[raw_h_idx * w + raw_w_idx + 1 * h * w];
            out[out_idx + 2 * out_h * out_w] = input[raw_h_idx * w + raw_w_idx + 2 * h * w];
        }
    } else {
        if (w_idx < t_w && h_idx < t_h)
        {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f;
            float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f;
            int top = raw_h_idx;
            int left = raw_w_idx;
            int bottom = MIN(h - 1, top + 1);
            int right = MIN(w - 1, left + 1);
            float y_diff = raw_h_idx - top;
            uint8_t a, b, c, d, out_tmp;
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            a = input[top * w + left + 0 * h * w];
            b = input[top * w + right + 0 * h * w];
            c = input[bottom * w + left + 0 * h * w];
            d = input[bottom * w + right + 0 * h * w];
            int out_idx = (h_idx + b_h) * out_w + b_w + w_idx;
            out_tmp = __float2int_rn(a * scale1 + b * scale2 + c * scale3 + d * scale4);
            out[out_idx + 0 * out_h * out_w] = out_tmp;

            a = input[top * w + left + 1 * h * w];
            b = input[top * w + right + 1 * h * w];
            c = input[bottom * w + left + 1 * h * w];
            d = input[bottom * w + right + 1 * h * w];
            out_tmp = __float2int_rn(a * scale1 + b * scale2 + c * scale3 + d * scale4);
            out[out_idx + 1 * out_h * out_w] = out_tmp;

            a = input[top * w + left + 2 * h * w];
            b = input[top * w + right + 2 * h * w];
            c = input[bottom * w + left + 2 * h * w];
            d = input[bottom * w + right + 2 * h * w];
            out_tmp = __float2int_rn(a * scale1 + b * scale2 + c * scale3 + d * scale4);
            out[out_idx + 2 * out_h * out_w] = out_tmp;
        }
    }
}

template<bool isBilinear = true>
__global__  void rgb_resize_ROI_kernel(const uchar3 *__restrict__ input,
        int h, int w, int h_out, int w_out, float t_h_trans, float t_w_trans,
        int roi_h_start, int roi_w_start, int h_roi, int w_roi, uchar3 *__restrict__ out) {
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (!isBilinear) {
        if (w_idx < w_out && h_idx < h_out)
        {
            int raw_h_idx = h_idx * t_h_trans + roi_h_start;
            int raw_w_idx = w_idx * t_w_trans + roi_w_start;
            out[h_idx * w_out + w_idx] = input[raw_h_idx * w + raw_w_idx];
        }
    } else {
        if (w_idx < w_out && h_idx < h_out)
        {
            float raw_h_idx = (h_idx + 0.5f) * t_h_trans - 0.5f + roi_h_start;
            float raw_w_idx = (w_idx + 0.5f) * t_w_trans - 0.5f + roi_w_start;
            int top = raw_h_idx;
            int left = raw_w_idx;
            int bottom = MIN(h - 1, top + 1);
            int right = MIN(w - 1, left + 1);
            uchar3 a, b, c, d, out_tmp;
            a = input[top * w + left];
            b = input[top * w + right];
            c = input[bottom * w + left];
            d = input[bottom * w + right];
            float y_diff = raw_h_idx - top;
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;

            out_tmp.x = __float2int_rn(a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4);
            out_tmp.y = __float2int_rn(a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4);
            out_tmp.z = __float2int_rn(a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4);
            out[h_idx * w_out + w_idx] = out_tmp;
        }
    }
}
#endif

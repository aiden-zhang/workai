#ifndef RGB_ROI_RESIZE_NORM_H_
#define RGB_ROI_RESIZE_NORM_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"


template<bool isBilinear = true, int channel_rev = 0>
__global__  void rgb_resize_ROI_norm_kernel(const uchar3 *__restrict__ input,
        float *__restrict__ out,
        int h, int w, int h_out, int w_out, int img_h, int img_w, int pad_h, int pad_w,
        float t_h_trans, float t_w_trans,
        int roi_h_start, int roi_w_start,
        int h_roi, int w_roi,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, 
        float pad1, float pad2, float pad3) {
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
    float3 norm = {pad1, pad2, pad3};
    if (!isBilinear) {
        int raw_h_idx = (h_idx - pad_h) * t_h_trans + roi_h_start;
        int in_offset = raw_h_idx * w + roi_w_start;
        if (w_idx < w_out) {
            if (h_run && w_run) {
                int raw_w_idx = (int)(w_idx * t_w_trans) + in_offset;
                uchar3 in_tmp = input[raw_w_idx];
                norm.x = (in_tmp.x * scale - mean1) * std1;
                norm.y = (in_tmp.y * scale - mean2) * std2;
                norm.z = (in_tmp.z * scale - mean3) * std3;
            }
            int out_idx = h_idx * w_out + w_idx;
            out[out_idx + channel_rev * w_out * h_out] = norm.x;
            out[out_idx + w_out * h_out] = norm.y;
            out[out_idx + (2 - channel_rev) * w_out * h_out] = norm.z;
        }
    } else {
        float raw_h_idx = (h_idx - pad_h + 0.5f) * t_h_trans - 0.5f + roi_h_start;
        int in_offset = raw_h_idx * w + roi_w_start;
        int top = raw_h_idx;
        int bottom = MIN(h - 1, top + 1);
        __half diff = __float2half_rn(raw_h_idx - top);
        top = MAX(0, top);
        int top_offset = top * w + roi_w_start;
        int bottom_offset = bottom * w + roi_w_start;
        
        if (w_idx < w_out) {
            if (h_run && w_run) {
                float raw_w_idx = (w_idx - pad_w + 0.5f) * t_w_trans - 0.5f;
                int left = raw_w_idx;
                int right = MIN(w - roi_w_start - 1, left + 1);
                half2 x1, x2;
                half2 w_bi, alpha_h;
                alpha_h.y = alpha_h.x = __float2half_rn(raw_w_idx - left);
                left = MAX(0, left);
                uchar3 a, b, c, d;
                a = input[top_offset + left];
                b = input[top_offset + right];
                c = input[bottom_offset + left];
                d = input[bottom_offset + right];
                x1.x = a.x;
                x2.x = b.x;
                x1.y = c.x;
                x2.y = d.x;
                
                w_bi = __hsub2(x2, x1);
                w_bi = __hfma2(w_bi, alpha_h, x1);
                int out_tmp = __half2int_rn((w_bi.y - w_bi.x) * diff + w_bi.x);
                norm.x = (out_tmp * scale - mean1) * std1;

                x1.x = a.y;
                x2.x = b.y;
                x1.y = c.y;
                x2.y = d.y;
                w_bi = __hsub2(x2, x1);
                w_bi = __hfma2(w_bi, alpha_h, x1);
                out_tmp = __half2int_rn((w_bi.y - w_bi.x) * diff + w_bi.x);
                norm.y = (out_tmp * scale - mean2) * std2;

                x1.x = a.z;
                x2.x = b.z;
                x1.y = c.z;
                x2.y = d.z;
                w_bi = __hsub2(x2, x1);
                w_bi = __hfma2(w_bi, alpha_h, x1);
                out_tmp = __half2int_rn((w_bi.y - w_bi.x) * diff + w_bi.x);
                norm.z = (out_tmp * scale - mean3) * std3;
            }
            int out_idx = h_idx * w_out + w_idx;
            out[out_idx + channel_rev * w_out * h_out] = norm.x;
            out[out_idx + w_out * h_out] = norm.y;
            out[out_idx + (2 - channel_rev) * w_out * h_out] = norm.z;
        }
    }
}

template<bool channel_rev = false>
__global__  void batch_rgb_resize_norm_kernel(uint8_t *__restrict__ input,
        float *__restrict__ out, int input_batch_offset, int output_batch_offset,
        batch_resize_param *param,
        float scale, float mean1, float mean2, float mean3,
        float std1, float std2, float std3, 
        float pad1, float pad2, float pad3) {
    input += blockIdx.y * input_batch_offset;
    out += blockIdx.y * output_batch_offset;
    __shared__ uint8_t sm[128];
    batch_resize_param *p = (batch_resize_param*)sm; // sizeof(batch_resize_param) = 24
    float *f = (float*)(p + 1); // 9 * sizeof(float) = 36
    if (threadIdx.x == 0) {
        p[0] = param[blockIdx.y];
        f[0] = pad1;
        f[1] = pad2;
        f[2] = pad3;
        f[3] = mean1;
        f[4] = mean2;
        f[5] = mean3;
        f[6] = std1;
        f[7] = std2;
        f[8] = std3;
    }
    __syncthreads();
    int in_h = p[0].in_h;
    int in_w = p[0].in_w;
    int out_h = p[0].out_h;
    int out_w = p[0].out_w;
    int img_h = p[0].img_h;
    int img_w = p[0].img_w;
    int pad_w = p[0].pad_w;
    int pad_h = p[0].pad_h;
    float w_scale = p[0].w_scale;
    float h_scale = p[0].h_scale;
    uchar3 *data = (uchar3*)input;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (; i < out_w * out_h; i += blockDim.x * gridDim.x) {
        int w_idx = i % out_w;
        int h_idx = i / out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm = {pad1, pad2, pad3};
        if (h_run && w_run) {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * h_scale - 0.5f;
            int top_h = raw_h_idx;
            int bottom_h = MIN(in_h - 1, top_h + 1);
            float y_diff = raw_h_idx - top_h;
            top_h = MAX(top_h, 0);
            float raw_w_idx = (w_idx - pad_w + 0.5f) * w_scale - 0.5f;
            int left_w = raw_w_idx;
            int right_w = MIN(in_w - 1, left_w + 1);
            float x_diff = raw_w_idx - left_w;
            left_w = MAX(left_w, 0);
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            uchar3 a = data[top_h * in_w + left_w];
            uchar3 b = data[top_h * in_w + right_w];
            uchar3 c = data[bottom_h * in_w + left_w];
            uchar3 d = data[bottom_h * in_w + right_w];
            int3 out_tmp;
            out_tmp.x = __half2int_rn(__half(a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4));
            out_tmp.y = __half2int_rn(__half(a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4));
            out_tmp.z = __half2int_rn(__half(a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4));
            norm.x = (out_tmp.x * scale - mean1) * std1;
            norm.y = (out_tmp.y * scale - mean2) * std2;
            norm.z = (out_tmp.z * scale - mean3) * std3;
        }
        if (channel_rev) {
            out[i + 2 * out_w * out_h] = norm.x;
            out[i + 1 * out_w * out_h] = norm.y;
            out[i + 0 * out_w * out_h] = norm.z;
        } else {
            out[i + 0 * out_w * out_h] = norm.x;
            out[i + 1 * out_w * out_h] = norm.y;
            out[i + 2 * out_w * out_h] = norm.z;
        }
    }
}

template <bool channel_rev = false>
__global__ void multi_roi_rgb_resize_norm_kernel(uint8_t *__restrict__ input,
                                                 float *__restrict__ out, int in_h, int in_w, int output_batch_offset,
                                                 multi_roi_resize_param *param,
                                                 float scale, float mean1, float mean2, float mean3,
                                                 float std1, float std2, float std3,
                                                 float pad1, float pad2, float pad3)
{
    out += blockIdx.y * output_batch_offset;
    __shared__ uint8_t sm[128];
    multi_roi_resize_param *p = (multi_roi_resize_param *)sm; // sizeof(multi_roi_resize_param) = 24
    if (threadIdx.x == 0)
    {
        p[0] = param[blockIdx.y];
    }
    __syncthreads();
    int roi_h_start = p[0].roi_h_start;
    int roi_w_start = p[0].roi_w_start;
    int roi_h = p[0].roi_h;
    int roi_w = p[0].roi_w;
    int out_h = p[0].out_h;
    int out_w = p[0].out_w;
    int img_h = p[0].img_h;
    int img_w = p[0].img_w;
    int pad_w = p[0].pad_w;
    int pad_h = p[0].pad_h;
    float w_scale = p[0].w_scale;
    float h_scale = p[0].h_scale;
    uchar3 *data = (uchar3 *)input;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < out_w * out_h; i += blockDim.x * gridDim.x)
    {
        int w_idx = i % out_w;
        int h_idx = i / out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm = {pad1, pad2, pad3};
        if (h_run && w_run)
        {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * h_scale - 0.5f + roi_h_start;
            int top_h = floorf(raw_h_idx);
            int bottom_h = MIN(roi_h + roi_h_start - 1, top_h + 1);
            float y_diff = raw_h_idx - top_h;
            float raw_w_idx = (w_idx - pad_w + 0.5f) * w_scale - 0.5f + roi_w_start;
            int left_w = floorf(raw_w_idx);
            int right_w = MIN(roi_w + roi_w_start - 1, left_w + 1);
            float x_diff = raw_w_idx - left_w;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            if (top_h < 0)
                top_h = 0;
            if (left_w < 0)
                left_w = 0;
            uchar3 a = data[top_h * in_w + left_w];
            uchar3 b = data[top_h * in_w + right_w];
            uchar3 c = data[bottom_h * in_w + left_w];
            uchar3 d = data[bottom_h * in_w + right_w];
            int3 out_tmp;
            out_tmp.x = __float2int_rn(a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4);
            out_tmp.y = __float2int_rn(a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4);
            out_tmp.z = __float2int_rn(a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4);
            norm.x = (out_tmp.x * scale - mean1) * std1;
            norm.y = (out_tmp.y * scale - mean2) * std2;
            norm.z = (out_tmp.z * scale - mean3) * std3;
        }
        if (channel_rev)
        {
            out[i + 2 * out_w * out_h] = norm.x;
            out[i + 1 * out_w * out_h] = norm.y;
            out[i + 0 * out_w * out_h] = norm.z;
        }
        else
        {
            out[i + 0 * out_w * out_h] = norm.x;
            out[i + 1 * out_w * out_h] = norm.y;
            out[i + 2 * out_w * out_h] = norm.z;
        }
    }
}
#endif

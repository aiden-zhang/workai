#ifndef YUV2RGB_RESIZE_NORM_H_
#define YUV2RGB_RESIZE_NORM_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"


__device__ __forceinline__ float clip(float value) {
  value += 0.5f;
  value = min(max(0.f,value),255.f);
  return value;
}

__device__ __forceinline__ float3 convert2rgb_TV_range(float y, float u, float v) {
    float3 tmp;
    y -= 16.f;
    y = max(0.f, y);
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(1.164f * y + 1.596f * v);
    tmp.y = clip(1.164f * y - 0.813f * v - 0.391f * u);
    tmp.z = clip(1.164f * y + 2.018f * u);
    return tmp;
}

__device__ __forceinline__ float3 convert2rgb_full_range(float y, float u, float v) {
    float3 tmp;
    u -= 128.f;
    v -= 128.f;
    tmp.x = clip(y + 1.403f * v);
    tmp.y = clip(y - 0.344f * u - 0.714f * v);
    tmp.z = clip(y + 1.773f * u);
    return tmp;
}

template <bool bgr_format = false, bool full_range = false>
__global__ void nv122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int top = raw_h_idx;
    int bottom = MIN(in_h - 1, top + 1);
    float y_diff = raw_h_idx - top;
    int in_y_offset0 = top * in_w;
    int in_y_offset1 = bottom * in_w;
    int in_uv_offset0 = in_w * in_h + (top >> 1) * in_w;
    int in_uv_offset1 = in_w * in_h + (bottom >> 1) * in_w;
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(w_idx < out_w) {
        int out_idx = h_idx * out_w + w_idx;
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = in[in_y_offset0 + left];
            b.x = in[in_y_offset0 + right];
            c.x = in[in_y_offset1 + left];
            d.x = in[in_y_offset1 + right];
            left = left & 0xfffffffe;
            right = right & 0xfffffffe;
            a.y = in[in_uv_offset0 + left];
            b.y = in[in_uv_offset0 + right];
            c.y = in[in_uv_offset1 + left];
            d.y = in[in_uv_offset1 + right];
            left++, right++;
            a.z = in[in_uv_offset0 + left];
            b.z = in[in_uv_offset0 + right];
            c.z = in[in_uv_offset1 + left];
            d.z = in[in_uv_offset1 + right];

            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }
            uchar3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void yu122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int top = raw_h_idx;
    int bottom = MIN(in_h - 1, top + 1);
    float y_diff = raw_h_idx - top;
    int in_y_offset0 = top * in_w;
    int in_y_offset1 = bottom * in_w;
    int in_uv_offset0 = (top >> 1) * (in_w >> 1);
    int in_uv_offset1 = (bottom >> 1) * (in_w >> 1);
    uint8_t *y = in;
    uint8_t *u = y + in_w * in_h;
    uint8_t *v = u + (in_w * in_h >> 2);
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (w_idx < out_w) {
        int out_idx = h_idx * out_w + w_idx;
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = y[in_y_offset0 + left];
            b.x = y[in_y_offset0 + right];
            c.x = y[in_y_offset1 + left];
            d.x = y[in_y_offset1 + right];
            left = left >> 1;
            right = right >> 1;
            a.y = u[in_uv_offset0 + left];
            b.y = u[in_uv_offset0 + right];
            c.y = u[in_uv_offset1 + left];
            d.y = u[in_uv_offset1 + right];
            a.z = v[in_uv_offset0 + left];
            b.z = v[in_uv_offset0 + right];
            c.z = v[in_uv_offset1 + left];
            d.z = v[in_uv_offset1 + right];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }

            uchar3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
    }
}


template <bool bgr_format = false, bool full_range = false>
__global__ void nv122rgb_nearest_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
        int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    int raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int in_y_offset = raw_h_idx * in_w;
    int in_uv_offset = in_w * in_h + (raw_h_idx >> 1) * in_w;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < out_w) {
        int out_idx = h_idx * out_w + w_idx;
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm_tmp = {pad1, pad2, pad3};
        if (h_run && w_run) {
            int raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            float3 yuv;
            yuv.x = in[in_y_offset + raw_w_idx];
            raw_w_idx = raw_w_idx & 0xfffffffe;
            yuv.y = in[in_uv_offset + raw_w_idx];
            yuv.z = in[in_uv_offset + raw_w_idx + 1];
            float3 out_tmp;
            if (full_range) {
                out_tmp = convert2rgb_full_range(yuv.x, yuv.y, yuv.z);
            } else {
                out_tmp = convert2rgb_TV_range(yuv.x, yuv.y, yuv.z);
            }
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void yu122rgb_nearest_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
        int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    int raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int in_y_offset = raw_h_idx * in_w;
    int in_uv_offset = (raw_h_idx >> 1) * (in_w >> 1);
    uint8_t *y = in;
    uint8_t *u = y + in_w * in_h;
    uint8_t *v = u + (in_w * in_h >> 2);
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < out_w) {
        int out_idx = h_idx * out_w + w_idx;
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            int raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            float3 yuv;
            yuv.x = y[in_y_offset + raw_w_idx];
            raw_w_idx = raw_w_idx >> 1;
            yuv.y = u[in_uv_offset + raw_w_idx];
            yuv.z = v[in_uv_offset + raw_w_idx];
            float3 out_tmp;
            if (full_range) {
                out_tmp = convert2rgb_full_range(yuv.x, yuv.y, yuv.z);
            } else {
                out_tmp = convert2rgb_TV_range(yuv.x, yuv.y, yuv.z);
            }
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + 2 * out_w * out_h] = norm_tmp.z;
    }
}

template <bool bgr_format = false, bool full_range = false, bool norm = true>
__global__ void roi_nv122rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int h_idx = blockIdx.y; 
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
    int top = raw_h_idx;
    int bottom = MIN(in_h - 1, top + 1);
    float y_diff = raw_h_idx - top;
    int in_y_offset0 = top * in_w + roi_w_start;
    int in_y_offset1 = bottom * in_w + roi_w_start;
    int in_uv_offset0 = in_w * in_h + (top >> 1) * in_w;
    int in_uv_offset1 = in_w * in_h + (bottom >> 1) * in_w;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < out_w) {
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            int left = raw_w_idx;
            int right = MIN(roi_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = in[in_y_offset0 + left]; 
            b.x = in[in_y_offset0 + right]; 
            c.x = in[in_y_offset1 + left]; 
            d.x = in[in_y_offset1 + right]; 
            left = (left + roi_w_start) & 0xfffffffe;
            right = (right + roi_w_start) & 0xfffffffe;
            a.y = in[in_uv_offset0 + left];
            b.y = in[in_uv_offset0 + right];
            c.y = in[in_uv_offset1 + left];
            d.y = in[in_uv_offset1 + right];
            left++, right++;
            a.z = in[in_uv_offset0 + left];
            b.z = in[in_uv_offset0 + right];
            c.z = in[in_uv_offset1 + left];
            d.z = in[in_uv_offset1 + right];

            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }

            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                if (norm) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.z;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.x;
                }
            } else {
                if (norm) {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.x;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.z;
                }
            }
        }
        int out_idx = h_idx * out_w + w_idx;
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + out_w * out_h * 2] = norm_tmp.z;
    }
}


template <bool bgr_format = false, bool full_range = false>
__global__ void roi_yuv4222rgb_resize_norm_fuse_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    int roi_w_pad_left = roi_w_start & 1;
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
    int top = raw_h_idx;
    int bottom = MIN(in_h - 1, top + 1);
    float y_diff = raw_h_idx - top;
    int top_offset = (top * in_w + roi_w_start - roi_w_pad_left) * 2;
    int bottom_offset = (bottom * in_w + roi_w_start - roi_w_pad_left) * 2;
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < out_w) {
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            int left = raw_w_idx;
            int right = MIN(roi_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            int offset = (left + roi_w_pad_left) << 1;
            a.x = in[top_offset + offset];
            c.x = in[bottom_offset + offset];
            offset = (right + roi_w_pad_left) << 1;
            b.x = in[top_offset + offset];
            d.x = in[bottom_offset + offset];
            offset = ((left + roi_w_pad_left) & 0xfffffffe)*2 + 1;
            a.y = in[top_offset + offset];
            a.z = in[top_offset + offset + 2];
            c.y = in[bottom_offset + offset];
            c.z = in[bottom_offset + offset + 2];
            offset = ((right + roi_w_pad_left) & 0xfffffffe)*2 + 1;
            b.y = in[top_offset + offset];
            b.z = in[top_offset + offset + 2];
            d.y = in[bottom_offset + offset];
            d.z = in[bottom_offset + offset + 2];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }
            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        int out_idx = h_idx * out_w + w_idx;
        out[out_idx] = norm_tmp.x;
        out[out_idx + out_w * out_h] = norm_tmp.y;
        out[out_idx + out_w * out_h * 2] = norm_tmp.z;
    }
}

template <bool full_range = false>
__global__ void roi_nv122rgba_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out,
        int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h) {
    int h_idx = blockIdx.y; 
    int in_y_offset = (h_idx + roi_h_start) * in_w + roi_w_start;
    int in_uv_offset = in_w * in_h + ((h_idx + roi_h_start) >> 1) * in_w + (roi_w_start & 0xfffffffe); 
    int w_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (w_idx < roi_w) {
        int uv_w_idx = (w_idx + (roi_w_start & 1)) & 0xfffffffe;
        float3 a;
        a.x = in[in_y_offset + w_idx];
        a.y = in[in_uv_offset + uv_w_idx];
        a.z = in[in_uv_offset + uv_w_idx + 1];

        if (full_range) {
            a = convert2rgb_full_range(a.x, a.y, a.z);
        } else {
            a = convert2rgb_TV_range(a.x, a.y, a.z);
        }
        uchar4 out_tmp;
        out_tmp.x = a.x;
        out_tmp.y = a.y;
        out_tmp.z = a.z;
        int out_idx = h_idx * roi_w + w_idx;
        out[out_idx] = out_tmp;
    }
}


template <bool full_range = false>
__global__ void roi_yu122rgba_kernel(uint8_t* __restrict__ in, uchar4* __restrict__ out, 
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h) {
    int h_idx = blockIdx.y;
    int half_w = in_w >> 1;
    uint8_t *in_u = in + in_w * in_h;
    uint8_t *in_v = in_u + in_w * in_h / 4;
    int uv_h1 = (h_idx + roi_h_start) >> 1;
    int in_y_offset = (h_idx + roi_h_start) * in_w + roi_w_start;
    int in_uv_offset = uv_h1 * half_w;
    for (int w_idx = threadIdx.x; w_idx < roi_w; w_idx += blockDim.x) {
        float3 a;
        a.x = in[in_y_offset + w_idx];
        int uv_w = (w_idx + roi_w_start)>>1;
        a.y = in_u[in_uv_offset + uv_w];
        a.z = in_v[in_uv_offset + uv_w];
        if (full_range) {
            a = convert2rgb_full_range(a.x, a.y, a.z);
        } else {
            a = convert2rgb_TV_range(a.x, a.y, a.z);
        }
        uchar4 out_tmp;
        out_tmp.x = a.x;
        out_tmp.y = a.y;
        out_tmp.z = a.z;
        int out_idx = h_idx * roi_w + w_idx;
        out[out_idx] = out_tmp;
    }
}

template <bool bgr_format = false, bool norm = true>
__global__ void rgba_resize_norm_quantize_fuse_kernel(uchar4* __restrict__ in, uint8_t* __restrict__ out,
        int in_w, int in_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
        float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
        float y_ratio,float x_ratio, float pad1, float pad2, float pad3, float zero_point, float scales_input) {
    int h_idx = blockIdx.y;
    bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
    float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f;
    int top = raw_h_idx;
    int bottom = MIN(in_h - 1, top + 1);
    float y_diff = raw_h_idx - top;
    int top_offset = top * in_w;
    int bottom_offset = bottom * in_w; 
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (w_idx < out_w) {
        float3 norm_tmp = {pad1, pad2, pad3};
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        if (h_run && w_run) {
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            uchar4 a, b, c, d;
            a = in[top_offset + left];
            b = in[top_offset + right];
            c = in[bottom_offset + left];
            d = in[bottom_offset + right];

            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                if (norm) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.z;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.x;
                }
            } else {
                if (norm) {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.x;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.z;
                }
            }
        }
        int out_idx = h_idx * out_w + w_idx;
        out[out_idx] = norm_tmp.x * scales_input + zero_point;
        out[out_idx + out_w * out_h] = norm_tmp.y * scales_input + zero_point;
        out[out_idx + (out_w << 1) * out_h] = norm_tmp.z * scales_input + zero_point;
    }
}

template <bool bgr_format = false, bool full_range = false, bool norm = true>
__global__ void roi_yu122rgb_resize_norm_general_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    uint8_t *y = in;
    uint8_t *u = in + in_w * in_h;
    uint8_t *v = in + in_w * in_h + (in_w * in_h >> 2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < out_h * out_w; idx += blockDim.x * gridDim.x) {
        int h_idx = idx / out_w;
        int w_idx = idx % out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm_tmp = {pad1, pad2, pad3};
        if (h_run && w_run) {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
            int top = raw_h_idx;
            int bottom = MIN(in_h - 1, top + 1);
            float y_diff = raw_h_idx - top;
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f + roi_w_start;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = y[top * in_w + left];
            b.x = y[top * in_w + right];
            c.x = y[bottom * in_w + left];
            d.x = y[bottom * in_w + right];
            int half_w = in_w >> 1;
            top = top >> 1;
            bottom = bottom >> 1;
            left = left >> 1;
            right = right >> 1;
            a.y = u[top * half_w + left];
            a.z = v[top * half_w + left];
            b.y = u[top * half_w + right];
            b.z = v[top * half_w + right];
            c.y = u[bottom * half_w + left];
            c.z = v[bottom * half_w + left];
            d.y = u[bottom * half_w + right];
            d.z = v[bottom * half_w + right];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }
            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                if (norm) {
                    norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.z;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.x;
                }
            } else {
                if (norm) {
                    norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                    norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                    norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
                } else {
                    norm_tmp.x = out_tmp.x;
                    norm_tmp.y = out_tmp.y;
                    norm_tmp.z = out_tmp.z;
                }
            }
        }
        out[idx] = norm_tmp.x;
        out[idx + out_w * out_h] = norm_tmp.y;
        out[idx + out_w * out_h * 2] = norm_tmp.z;
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void roi_yuv444p2rgb_resize_norm_general_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    uint8_t *y = in;
    uint8_t *u = in + in_w * in_h;
    uint8_t *v = in + in_w * in_h * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < out_h * out_w; idx += blockDim.x * gridDim.x) {
        int h_idx = idx / out_w;
        int w_idx = idx % out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm_tmp = {pad1, pad2, pad3};
        if (h_run && w_run) {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
            int top = raw_h_idx;
            int bottom = MIN(in_h - 1, top + 1);
            float y_diff = raw_h_idx - top;
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f + roi_w_start;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = y[top * in_w + left];
            a.y = u[top * in_w + left];
            a.z = v[top * in_w + left];
            b.x = y[top * in_w + right];
            b.y = u[top * in_w + right];
            b.z = v[top * in_w + right];
            c.x = y[bottom * in_w + left];
            c.y = u[bottom * in_w + left];
            c.z = v[bottom * in_w + left];
            d.x = y[bottom * in_w + right];
            d.y = u[bottom * in_w + right];
            d.z = v[bottom * in_w + right];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }
            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[idx] = norm_tmp.x;
        out[idx + out_w * out_h] = norm_tmp.y;
        out[idx + out_w * out_h * 2] = norm_tmp.z;
    }
}

__global__ void roi_yuv400p2rgb_resize_norm_general_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean, float std, float scale, float y_ratio,float x_ratio, float pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < out_h * out_w; idx += blockDim.x * gridDim.x) {
        int h_idx = idx / out_w;
        int w_idx = idx % out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float norm_tmp = pad;
        if (h_run && w_run) {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
            int top = raw_h_idx;
            int bottom = MIN(in_h - 1, top + 1);
            float y_diff = raw_h_idx - top;
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f + roi_w_start;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float a, b, c, d;
            a = in[top * in_w + left];
            b = in[top * in_w + right];
            c = in[bottom * in_w + left];
            d = in[bottom * in_w + right];
            float out_tmp = a * scale1 + b * scale2 + c * scale3 + d * scale4;
            norm_tmp = (out_tmp * scale - mean) * std;
        }
        out[idx] = norm_tmp;
    }
}

template <bool bgr_format = false, bool full_range = false>
__global__ void roi_yuv422p2rgb_resize_norm_general_kernel(uint8_t* __restrict__ in, float* __restrict__ out,
    int in_w, int in_h, int roi_w_start, int roi_h_start, int roi_w, int roi_h, int img_w, int img_h, int out_w, int out_h, int pad_w, int pad_h,
    float mean1, float mean2, float mean3, float std1, float std2, float std3, float scale,
    float y_ratio,float x_ratio, float pad1, float pad2, float pad3) {
    uint8_t *y = in;
    uint8_t *u = in + in_w * in_h;
    uint8_t *v = in + in_w * in_h + (in_w * in_h >> 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < out_h * out_w; idx += blockDim.x * gridDim.x) {
        int h_idx = idx / out_w;
        int w_idx = idx % out_w;
        bool h_run = h_idx >= pad_h && h_idx < (pad_h + img_h);
        bool w_run = w_idx >= pad_w && w_idx < (pad_w + img_w);
        float3 norm_tmp = {pad1, pad2, pad3};
        if (h_run && w_run) {
            float raw_h_idx = (h_idx - pad_h + 0.5f) * y_ratio - 0.5f + roi_h_start;
            int top = raw_h_idx;
            int bottom = MIN(in_h - 1, top + 1);
            float y_diff = raw_h_idx - top;
            float raw_w_idx = (w_idx - pad_w + 0.5f) * x_ratio - 0.5f + roi_w_start;
            int left = raw_w_idx;
            int right = MIN(in_w - 1, left + 1);
            float x_diff = raw_w_idx - left;
            float scale1 = (1.f - x_diff) * (1.f - y_diff);
            float scale2 = x_diff * (1.f - y_diff);
            float scale3 = (1.f - x_diff) * y_diff;
            float scale4 = x_diff * y_diff;
            float3 a, b, c, d;
            a.x = y[top * in_w + left];
            b.x = y[top * in_w + right];
            c.x = y[bottom * in_w + left];
            d.x = y[bottom * in_w + right];
            left = left >> 1;
            right = right >> 1;
            int half_w = in_w >> 1;
            a.y = u[top * half_w + left];
            a.z = v[top * half_w + left];
            b.y = u[top * half_w + right];
            b.z = v[top * half_w + right];
            c.y = u[bottom * half_w + left];
            c.z = v[bottom * half_w + left];
            d.y = u[bottom * half_w + right];
            d.z = v[bottom * half_w + right];
            if (full_range) {
                a = convert2rgb_full_range(a.x, a.y, a.z);
                b = convert2rgb_full_range(b.x, b.y, b.z);
                c = convert2rgb_full_range(c.x, c.y, c.z);
                d = convert2rgb_full_range(d.x, d.y, d.z);
            } else {
                a = convert2rgb_TV_range(a.x, a.y, a.z);
                b = convert2rgb_TV_range(b.x, b.y, b.z);
                c = convert2rgb_TV_range(c.x, c.y, c.z);
                d = convert2rgb_TV_range(d.x, d.y, d.z);
            }
            float3 out_tmp;
            out_tmp.x = a.x * scale1 + b.x * scale2 + c.x * scale3 + d.x * scale4;
            out_tmp.y = a.y * scale1 + b.y * scale2 + c.y * scale3 + d.y * scale4;
            out_tmp.z = a.z * scale1 + b.z * scale2 + c.z * scale3 + d.z * scale4;
            if (bgr_format) {
                norm_tmp.x = (out_tmp.z * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.x * scale - mean3) * std3;
            } else {
                norm_tmp.x = (out_tmp.x * scale - mean1) * std1;
                norm_tmp.y = (out_tmp.y * scale - mean2) * std2;
                norm_tmp.z = (out_tmp.z * scale - mean3) * std3;
            }
        }
        out[idx] = norm_tmp.x;
        out[idx + out_w * out_h] = norm_tmp.y;
        out[idx + out_w * out_h * 2] = norm_tmp.z;
    }
}
#endif

#ifndef YUV_RESIZE_H_
#define YUV_RESIZE_H_

#include <stdio.h>
#include "cuda_fp16.h"
#include "common.h"

__global__ void nearest_resize_kernel_nv12_nv21(const uint8_t *__restrict__ in_Y,
                                                const uint8_t *__restrict__ in_UV,
                                                uint8_t *__restrict__ out_Y,
                                                uint8_t *__restrict__ out_UV,
                                                int h, int w, int t_h, int t_w,
                                                float t_h_trans, float t_w_trans)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < t_w * (t_h >> 1))
    {
        int h_idx = idx / t_w;
        int w_idx = idx % t_w;

        int dst_h_idx = h_idx << 1;
        int src_h_idx1 = dst_h_idx * t_h_trans;
        int src_h_idx2  = (dst_h_idx + 1) * t_h_trans;
        int uv_src_h_idx = src_h_idx1 >> 1;
        int src_w_idx = w_idx * t_w_trans;
        int out_offset = dst_h_idx * t_w + w_idx;

        out_Y[out_offset] = in_Y[src_h_idx1 * w + src_w_idx];
        out_Y[out_offset + t_w] = in_Y[src_h_idx2 * w + src_w_idx];

        if ((w_idx & 0x1) == 0)
        {
            int uv_in_offset = src_w_idx >> 1;
            uv_in_offset = w * (src_h_idx1 >> 1) + uv_in_offset * 2;
            int uv_offset = (h_idx * (t_w >> 1) + (w_idx >> 1)) * 2;
            out_UV[uv_offset + 0] = in_UV[uv_in_offset + 0];
            out_UV[uv_offset + 1] = in_UV[uv_in_offset + 1];
        }
    }
}

__global__ void nearest_resize_kernel_i420(const uint8_t *__restrict__ in_Y,
                                           const uint8_t *__restrict__ in_U,
                                           const uint8_t *__restrict__ in_V,
                                           uint8_t *__restrict__ out_Y,
                                           uint8_t *__restrict__ out_U,
                                           uint8_t *__restrict__ out_V,
                                           int h, int w, int t_h, int t_w,
                                           float t_h_trans, float t_w_trans)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < t_w * (t_h >> 1))
    {
        int h_idx = idx / t_w;
        int w_idx = idx % t_w;

        int dst_h_idx = h_idx << 1;
        int src_h_idx1 = dst_h_idx * t_h_trans;
        int src_h_idx2  = (dst_h_idx + 1) * t_h_trans;
        int src_w_idx = w_idx * t_w_trans;
        int out_offset = dst_h_idx * t_w + w_idx;

        out_Y[out_offset] = in_Y[src_h_idx1 * w + src_w_idx];       //[0,0]
        out_Y[out_offset + t_w] = in_Y[src_h_idx2 * w + src_w_idx]; //[1,0]

        if ((w_idx & 0x1) == 0)
        {
            int uv_in_offset = (src_w_idx >> 1) + (w >> 1) * (src_h_idx1 >> 1);
            int uv_offset = h_idx * (t_w >> 1) + (w_idx >> 1);
            out_U[uv_offset] = in_U[uv_in_offset];
            out_V[uv_offset] = in_V[uv_in_offset];
        }
    }
}

__global__ void bilinear_resize_kernel_nv12_nv21(const uint8_t *__restrict__ in_Y,
                                                 const uint8_t *__restrict__ in_UV,
                                                 uint8_t *__restrict__ out_Y,
                                                 uint8_t *__restrict__ out_UV,
                                                 int h, int w, int t_h, int t_w,
                                                 float t_h_trans, float t_w_trans)
{
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (w_idx < (t_w >> 1) && h_idx < (t_h >> 1))
    {
        int dst_h_idx = h_idx << 1;
        float src_h_idx = dst_h_idx * t_h_trans;
        int top = src_h_idx;
        int bottom  = MIN(h - 1, top + 1);
        __half dis = __float2half_rn(src_h_idx - top);
        int dst_w_idx = w_idx << 1;
        float src_w_idx = dst_w_idx * t_w_trans;
        int left   = src_w_idx;
        int right  = MIN(w - 1, left + 1);
        half2 w_bi, alpha_h;
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        half2 x1, x2;
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        int out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        int out_uv_offset = h_idx * (t_w >> 1) + w_idx;
        left = left & 0xfffffffe;
        right = right & 0xfffffffe;
        int offset1 = w * (top >> 1);
        int offset2 = w * (bottom >> 1);
        x1.x = in_UV[offset1 + left];
        x2.x = in_UV[offset1 + right];
        x1.y = in_UV[offset2 + left];
        x2.y = in_UV[offset2 + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_UV[out_uv_offset * 2 + 0] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);
        left++;
        right++;
        x1.x = in_UV[offset1 + left];
        x2.x = in_UV[offset1 + right];
        x1.y = in_UV[offset2 + left];
        x2.y = in_UV[offset2 + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_UV[out_uv_offset * 2 + 1] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_w_idx++;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_h_idx++;
        src_h_idx = dst_h_idx * t_h_trans;
        top = src_h_idx;
        bottom = MIN(h - 1, top + 1);
        dis = __float2half_rn(src_h_idx - top);
        dst_w_idx = w_idx << 1;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_w_idx++;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);
    }
}

__global__ void bilinear_resize_kernel_i420(const uint8_t *__restrict__ in_Y,
                                            const uint8_t *__restrict__ in_U,
                                            const uint8_t *__restrict__ in_V,
                                            uint8_t *__restrict__ out_Y,
                                            uint8_t *__restrict__ out_U,
                                            uint8_t *__restrict__ out_V,
                                            int h, int w, int t_h, int t_w,
                                            float t_h_trans, float t_w_trans)
{
    int w_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_idx = threadIdx.y + blockIdx.y * blockDim.y;
    if (w_idx < (t_w >> 1) && h_idx < (t_h >> 1))
    {
        int dst_h_idx = h_idx << 1;
        float src_h_idx = dst_h_idx * t_h_trans;
        int top = src_h_idx;
        int bottom = MIN(h - 1, top + 1);
        __half dis = __float2half_rn(src_h_idx - top);
        int dst_w_idx = w_idx << 1;
        float src_w_idx = dst_w_idx * t_w_trans;
        int left = src_w_idx;
        int right = MIN(w - 1, left + 1);
        half2 w_bi, alpha_h;
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        half2 x1, x2;
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        int out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        int uv_offset = h_idx * (t_w >> 1) + w_idx;
        left = left >> 1;
        right = right >> 1;
        int offset1 = (w >> 1) * (top >> 1);
        int offset2 = (w >> 1) * (bottom >> 1);
        x1.x = in_U[offset1 + left];
        x2.x = in_U[offset1 + right];
        x1.y = in_U[offset2 + left];
        x2.y = in_U[offset2 + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_U[uv_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);
        x1.x = in_V[offset1 + left];
        x2.x = in_V[offset1 + right];
        x1.y = in_V[offset2 + left];
        x2.y = in_V[offset2 + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_V[uv_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_w_idx++;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_h_idx++;
        src_h_idx = dst_h_idx * t_h_trans;
        top = src_h_idx;
        bottom = MIN(h - 1, top + 1);
        dis = __float2half_rn(src_h_idx - top);
        dst_w_idx = w_idx << 1;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);

        dst_w_idx++;
        src_w_idx = dst_w_idx * t_w_trans;
        left = src_w_idx;
        right = MIN(w - 1, left + 1);
        alpha_h.y = alpha_h.x = __float2half_rn(src_w_idx - left);
        x1.x = in_Y[top * w + left];
        x2.x = in_Y[top * w + right];
        x1.y = in_Y[bottom * w + left];
        x2.y = in_Y[bottom * w + right];
        w_bi = __hsub2(x2, x1);
        w_bi = __hfma2(w_bi, alpha_h, x1);
        out_offset = dst_h_idx * t_w + dst_w_idx;
        out_Y[out_offset] = __half2int_rn((w_bi.y - w_bi.x) * dis + w_bi.x);
    }
}

#endif
